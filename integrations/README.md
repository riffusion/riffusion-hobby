# Integrations

This package contains integrations of Riffusion into third party apps and deployments.

## Baseten

[Baseten](https://baseten.com) is a platform for building and deploying machine learning models.

## Replicate

To run riffusion as a Cog model, first, [install Cog](https://github.com/replicate/cog) and
download the model weights:

    cog run python -m integrations.cog_riffusion --download_weights

Then you can run predictions:

    cog predict -i prompt_a="funky synth solo"

You can also view the model on replicate [here](https://replicate.com/hmartiro/riffusion). Owners
can push an updated version of the model like so:

    cog push r8.im/hmartiro/riffusion
