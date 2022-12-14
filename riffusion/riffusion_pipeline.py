"""
Riffusion inference pipeline.
"""

import functools
import inspect
import typing as T

import numpy as np
import PIL
import torch

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from .datatypes import InferenceInput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RiffusionPipeline(DiffusionPipeline):
    """
    Diffusers pipeline for doing a controlled img2img interpolation for audio generation.

    # TODO(hayk): Document more

    Part of this code was adapted from the non-img2img interpolation pipeline at:

        https://github.com/huggingface/diffusers/blob/main/examples/community/interpolate_stable_diffusion.py

    Check the documentation for DiffusionPipeline for full information.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @functools.lru_cache()
    def embed_text(self, text):
        """
        Takes in text and turns it into text embeddings.
        """
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    @functools.lru_cache()
    def embed_text_weighted(self, text):
        """
        Get text embedding with weights.
        """
        from .prompt_weighting import get_weighted_text_embeddings

        return get_weighted_text_embeddings(
            pipe=self,
            prompt=text,
            uncond_prompt=None,
            max_embeddings_multiples=3,
            no_boseos_middle=False,
            skip_parsing=False,
            skip_weighting=False,
        )[0]

    @torch.no_grad()
    def riffuse(
        self,
        inputs: InferenceInput,
        init_image: PIL.Image.Image,
        mask_image: PIL.Image.Image = None,
        use_reweighting: bool = True,
    ) -> PIL.Image.Image:
        """
        Runs inference using interpolation with both img2img and text conditioning.

        Args:
            inputs: Parameter dataclass
            init_image: Image used for conditioning
            mask_image: White pixels in the mask will be replaced by noise and therefore repainted,
                        while black pixels will be preserved. It will be converted to a single
                        channel (luminance) before use.
            use_reweighting: Use prompt reweighting
        """
        alpha = inputs.alpha
        start = inputs.start
        end = inputs.end

        guidance_scale = start.guidance * (1.0 - alpha) + end.guidance * alpha
        generator_start = torch.Generator(device=self.device).manual_seed(start.seed)
        generator_end = torch.Generator(device=self.device).manual_seed(end.seed)

        # Text encodings
        if use_reweighting:
            embed_start = self.embed_text_weighted(start.prompt)
            embed_end = self.embed_text_weighted(end.prompt)
        else:
            embed_start = self.embed_text(start.prompt)
            embed_end = self.embed_text(end.prompt)
        text_embedding = torch.lerp(embed_start, embed_end, alpha)

        # Image latents
        init_image = preprocess_image(init_image)
        init_image_torch = init_image.to(device=self.device, dtype=embed_start.dtype)
        init_latent_dist = self.vae.encode(init_image_torch).latent_dist
        # TODO(hayk): Probably this seed should just be 0 always? Make it 100% symmetric. The
        # result is so close no matter the seed that it doesn't really add variety.
        generator = torch.Generator(device=self.device).manual_seed(start.seed)
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # Prepare mask latent
        if mask_image:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            mask_image = preprocess_mask(mask_image, scale_factor=vae_scale_factor)
            mask = mask_image.to(device=self.device, dtype=embed_start.dtype)
        else:
            mask = None

        outputs = self.interpolate_img2img(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            mask=mask,
            generator_a=generator_start,
            generator_b=generator_end,
            interpolate_alpha=alpha,
            strength_a=start.denoising,
            strength_b=end.denoising,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return outputs["images"][0]

    @torch.no_grad()
    def interpolate_img2img(
        self,
        text_embeddings: torch.FloatTensor,
        init_latents: torch.FloatTensor,
        generator_a: torch.Generator,
        generator_b: torch.Generator,
        interpolate_alpha: float,
        mask: T.Optional[torch.FloatTensor] = None,
        strength_a: float = 0.8,
        strength_b: float = 0.8,
        num_inference_steps: T.Optional[int] = 50,
        guidance_scale: T.Optional[float] = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_images_per_prompt: T.Optional[int] = 1,
        eta: T.Optional[float] = 0.0,
        output_type: T.Optional[str] = "pil",
        **kwargs,
    ):
        """
        TODO
        """
        batch_size = text_embeddings.shape[0]

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                uncond_tokens = negative_prompt

            # max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_images_per_prompt, dim=0
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype

        strength = (1 - interpolate_alpha) * strength_a + interpolate_alpha * strength_b

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # add noise to latents using the timesteps
        noise_a = torch.randn(
            init_latents.shape, generator=generator_a, device=self.device, dtype=latents_dtype
        )
        noise_b = torch.randn(
            init_latents.shape, generator=generator_b, device=self.device, dtype=latents_dtype
        )
        noise = slerp(interpolate_alpha, noise_a, noise_b)
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents.clone()

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                # import ipdb; ipdb.set_trace()
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)


def preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask: PIL.Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize(
        (w // scale_factor, h // scale_factor), resample=PIL.Image.NEAREST
    )
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)

    return mask


def slerp(t, v0, v1, dot_threshold=0.9995):
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2
