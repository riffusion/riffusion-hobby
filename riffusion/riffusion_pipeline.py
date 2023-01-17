"""
Riffusion inference pipeline.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import typing as T

import numpy as np
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from riffusion.datatypes import InferenceInput
from riffusion.external.prompt_weighting import get_weighted_text_embeddings
from riffusion.util import torch_util

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

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        use_traced_unet: bool = True,
        channels_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ) -> RiffusionPipeline:
        """
        Load the riffusion model pipeline.

        Args:
            checkpoint: Model checkpoint on disk in diffusers format
            use_traced_unet: Whether to use the traced unet for speedups
            device: Device to load the model on
            channels_last: Whether to use channels_last memory format
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        device = torch_util.check_device(device)

        if device == "cpu" or device.lower().startswith("mps"):
            print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
            dtype = torch.float32

        pipeline = RiffusionPipeline.from_pretrained(
            checkpoint,
            revision="main",
            torch_dtype=dtype,
            # Disable the NSFW filter, causes incorrect false positives
            # TODO(hayk): Disable the "you have passed a non-standard module" warning from this.
            safety_checker=lambda images, **kwargs: (images, False),
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(device)

        if channels_last:
            pipeline.unet.to(memory_format=torch.channels_last)

        # Optionally load a traced unet
        if checkpoint == "riffusion/riffusion-model-v1" and use_traced_unet:
            traced_unet = cls.load_traced_unet(
                checkpoint=checkpoint,
                subfolder="unet_traced",
                filename="unet_traced.pt",
                in_channels=pipeline.unet.in_channels,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )

            if traced_unet is not None:
                pipeline.unet = traced_unet

        model = pipeline.to(device)

        return model

    @staticmethod
    def load_traced_unet(
        checkpoint: str,
        subfolder: str,
        filename: str,
        in_channels: int,
        dtype: torch.dtype,
        device: str = "cuda",
        local_files_only=False,
        cache_dir: T.Optional[str] = None,
    ) -> T.Optional[torch.nn.Module]:
        """
        Load a traced unet from the huggingface hub. This can improve performance.
        """
        if device == "cpu" or device.lower().startswith("mps"):
            print("WARNING: Traced UNet only available for CUDA, skipping")
            return None

        # Download and load the traced unet
        unet_file = hf_hub_download(
            checkpoint,
            subfolder=subfolder,
            filename=filename,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        unet_traced = torch.jit.load(unet_file)

        # Wrap it in a torch module
        class TracedUNet(torch.nn.Module):
            @dataclasses.dataclass
            class UNet2DConditionOutput:
                sample: torch.FloatTensor

            def __init__(self):
                super().__init__()
                self.in_channels = device
                self.device = device
                self.dtype = dtype

            def forward(self, latent_model_input, t, encoder_hidden_states):
                sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                return self.UNet2DConditionOutput(sample=sample)

        return TracedUNet()

    @property
    def device(self) -> str:
        return str(self.vae.device)

    @functools.lru_cache()
    def embed_text(self, text) -> torch.FloatTensor:
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
    def embed_text_weighted(self, text) -> torch.FloatTensor:
        """
        Get text embedding with weights.
        """
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
        init_image: Image.Image,
        mask_image: T.Optional[Image.Image] = None,
        use_reweighting: bool = True,
    ) -> Image.Image:
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

        # TODO(hayk): Always generate the seed on CPU?
        if self.device.lower().startswith("mps"):
            generator_start = torch.Generator(device="cpu").manual_seed(start.seed)
            generator_end = torch.Generator(device="cpu").manual_seed(end.seed)
        else:
            generator_start = torch.Generator(device=self.device).manual_seed(start.seed)
            generator_end = torch.Generator(device=self.device).manual_seed(end.seed)

        # Text encodings
        if use_reweighting:
            embed_start = self.embed_text_weighted(start.prompt)
            embed_end = self.embed_text_weighted(end.prompt)
        else:
            embed_start = self.embed_text(start.prompt)
            embed_end = self.embed_text(end.prompt)

        text_embedding = embed_start + alpha * (embed_end - embed_start)

        # Image latents
        init_image_torch = preprocess_image(init_image).to(
            device=self.device, dtype=embed_start.dtype
        )
        init_latent_dist = self.vae.encode(init_image_torch).latent_dist
        # TODO(hayk): Probably this seed should just be 0 always? Make it 100% symmetric. The
        # result is so close no matter the seed that it doesn't really add variety.
        if self.device.lower().startswith("mps"):
            generator = torch.Generator(device="cpu").manual_seed(start.seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(start.seed)

        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # Prepare mask latent
        mask: T.Optional[torch.Tensor] = None
        if mask_image:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            mask = preprocess_mask(mask_image, scale_factor=vae_scale_factor).to(
                device=self.device, dtype=embed_start.dtype
            )

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
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        generator_a: torch.Generator,
        generator_b: torch.Generator,
        interpolate_alpha: float,
        mask: T.Optional[torch.Tensor] = None,
        strength_a: float = 0.8,
        strength_b: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_images_per_prompt: int = 1,
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
        noise = torch_util.slerp(interpolate_alpha, noise_a, noise_b)
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same args
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
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, torch.tensor([t])
                )
                # import ipdb; ipdb.set_trace()
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0


def preprocess_mask(mask: Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Tile and transpose
    mask_np = np.tile(mask_np, (4, 1, 1))
    mask_np = mask_np[None].transpose(0, 1, 2, 3)  # what does this step do?

    # Invert to repaint white and keep black
    mask_np = 1 - mask_np  # repaint white, keep black

    return torch.from_numpy(mask_np)
