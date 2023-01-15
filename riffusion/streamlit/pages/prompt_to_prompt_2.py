
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
