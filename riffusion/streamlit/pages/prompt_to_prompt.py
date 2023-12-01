import torch
import numpy as np
import streamlit as st
from PIL import Image
from tqdm import tqdm

from riffusion.external.null_text_inversion.invertible_ddim_scheduler import InvertibleDDIMScheduler
from riffusion.streamlit import util as streamlit_util


def show_lat(pipe, latents):
    # utility function for visualization of diffusion process
    with torch.no_grad():
        images = pipe.decode_latents(latents)
        im = pipe.numpy_to_pil(images)[0]#.resize((128, 128))
    return im

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def im2latent(pipe, im, generator):
    init_image = preprocess(im).to(pipe.device)
    init_latent_dist = pipe.vae.encode(init_image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)

    return init_latents * 0.18215

def render_prompt_to_prompt() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":pen: Prompt to Prompt")
    st.write(
        """
    TODO
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            TODO
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    # import sys

    # sys.path.append("/home/ubuntu/prompt-to-prompt")

    # import ptp_utils

    # photo from ffhq
    from pathlib import Path
    # repo root
    img_path = Path(__file__).parent.parent.parent.parent / "test" / "test_data" / "tired_traveler" / "images" / "clip_2_start_103694_ms_duration_5678_ms.png"
    init_image = Image.open(str(img_path)).resize((512,512))
    st.write("## Initial Image")
    st.image(init_image)

    pipeline = streamlit_util.load_stable_diffusion_pipeline(
        device=device,
        # TODO(hayk): Make float16 work
        dtype=torch.float32,
    )
    st.write(pipeline.scheduler.config)
    pipeline.scheduler = InvertibleDDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    st.write(pipeline.scheduler)
    pipe = pipeline

    # fix seed
    seed = 84
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    image_latents = im2latent(pipe, init_image, g)

    # img = show_lat(pipe, image_latents)
    # st.image(img, caption="Initial Latents to Image")

    pipe.scheduler.set_timesteps(51)

    # use text describing an image
    source_prompt = "electric guitar"
    context = pipe._encode_prompt(source_prompt, pipe.device, 1, False, "")

    st.write("## Adding Noise")
    decoded_latents = image_latents.clone()
    st.image(show_lat(pipe, decoded_latents), caption="timestep 0")
    with torch.autocast("cuda"), torch.inference_mode():
        # we are pivoting timesteps as we are moving in opposite direction
        timesteps = pipe.scheduler.timesteps.flip(0)
        # this would be our targets for pivoting
        init_trajectory = torch.empty(len(timesteps), *decoded_latents.size()[1:], device=decoded_latents.device, dtype=decoded_latents.dtype)
        for i, t in enumerate(tqdm(timesteps)):
            init_trajectory[i:i+1] = decoded_latents
            noise_pred = pipe.unet(decoded_latents, t, encoder_hidden_states=context).sample
            decoded_latents = pipe.scheduler.reverse_step(noise_pred, t, decoded_latents).next_sample
            if i < 10:
                st.image(show_lat(pipe, decoded_latents), caption=f"timestep {i + 1}")

    latents = decoded_latents.clone()
    st.write("## Removing Noise Naively")
    with torch.autocast("cuda"), torch.inference_mode():
        for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents = pipe.scheduler.step(
                pipe.unet(latents, t, encoder_hidden_states=context).sample, t, latents
            ).prev_sample
            if i > 40:
                st.image(show_lat(pipe, latents), caption=f"timestep {len(pipe.scheduler.timesteps) - i - 1}")

    return

    init_trajectory = init_trajectory.cpu().flip(0)
    _ = pipe.vae.requires_grad_(False)
    _ = pipe.text_encoder.requires_grad_(False)
    _ = pipe.unet.requires_grad_(False)

    latents = decoded_latents.clone()

    # I've noticed that scale < 1 works better
    scale = 0.6

    context_uncond = pipe._encode_prompt("", pipe.device, 1, False, "")
    # we will be optimizing uncond text embedding
    context_uncond.requires_grad_(True)

    # use same text
    prompt = source_prompt
    context_cond = pipe._encode_prompt(prompt, pipe.device, 1, False, "")

    # default lr works
    opt = torch.optim.AdamW([context_uncond])

    # concat latents for classifier-free guidance
    latents = torch.cat([latents, latents])
    latents.requires_grad_(True)
    context = torch.cat((context_uncond, context_cond))

    st.write("## Removing Noise with Pivoting")
    with torch.autocast(device):
        for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents = pipe.scheduler.scale_model_input(latents, t)
            uncond, cond = pipe.unet(latents, t, encoder_hidden_states=context).sample.chunk(2)
            with torch.enable_grad():
                latents = pipe.scheduler.step(uncond + scale * (cond - uncond), t, latents, generator=g).prev_sample

            opt.zero_grad()
            # optimize uncond text emb
            pivot_value = init_trajectory[[i]].to(pipe.device)
            (latents - pivot_value).mean().backward()
            opt.step()
            latents = latents.detach()

            if i % 10 == 0:
                st.image(show_lat(pipe, latents), caption=f"timestep {len(pipe.scheduler.timesteps) - i - 1}")

    st.write("## Removing Noise with Pivoting and Text")
    latents = decoded_latents.clone()

    # for image editing purposes scale from 1 to 2 works good
    scale = 1.5

    context_uncond = pipe._encode_prompt("", pipe.device, 1, False, "")
    # we will be optimizing uncond text embedding
    context_uncond.requires_grad_(True)

    # use same text
    prompt = "jazz saxophone"
    context_cond = pipe._encode_prompt(prompt, pipe.device, 1, False, "")

    # default lr works
    opt = torch.optim.AdamW([context_uncond])

    # concat latents for classifier-free guidance
    latents = torch.cat([latents, latents])
    latents.requires_grad_(True)
    context = torch.cat((context_uncond, context_cond))

    with torch.autocast(device):
        for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents = pipe.scheduler.scale_model_input(latents, t)
            uncond, cond = pipe.unet(latents, t, encoder_hidden_states=context).sample.chunk(2)
            with torch.enable_grad():
                latents = pipe.scheduler.step(uncond + scale * (cond - uncond), t, latents, generator=g).prev_sample

            opt.zero_grad()
            # optimize uncond text emb
            pivot_value = init_trajectory[[i]].to(pipe.device)
            (latents - pivot_value).mean().backward()
            opt.step()
            latents = latents.detach()

            if i % 10 == 0:
                st.image(show_lat(pipe, latents), caption=f"timestep {len(pipe.scheduler.timesteps) - i - 1}")

    # original_prompt = "electric guitar"
    # new_prompt = "piano"
    # prompts = [original_prompt, new_prompt]

    # controller = None
    # num_inference_steps = 50
    # guidance_scale = 7.5
    # generator = None

    # cross_replace_steps = {
    #     "default_": 0.8,
    # }
    # self_replace_steps = 0.6
    # blend_word = (("cat",), ("cat",))  # for local edit
    # eq_params = {
    #     "words": (
    #         "silver",
    #         "sculpture",
    #     ),
    #     "values": (
    #         2,
    #         2,
    #     ),
    # }  # amplify attention to the words "silver" and "sculpture" by *2

    # controller = make_controller(
    #     prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params
    # )

    # images, _ = ptp_utils.text2image_ldm_stable(
    #     model=pipeline,
    #     prompt=prompts,
    #     controller=controller,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    #     generator=generator,
    #     latent=None,
    #     low_resource=False,
    # )


if __name__ == "__main__":
    render_prompt_to_prompt()
