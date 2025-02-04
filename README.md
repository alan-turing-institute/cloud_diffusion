
# Cloud Diffusion for Solar Energy Forecasting

This codebase is adapted from the [Diffusion on the Clouds](https://github.com/capecape/diffusion-on-the-clouds) project for the purposes of generating short-term solar energy forecasts. It was developed as part of a research project with Open Climate Fix.

## Setup

1. Clone this repository and run `pip install -e .` or `pip install cloud_diffusion`
2. Set up your WandB account by signing up at [wandb.ai](https://wandb.ai/site).
3. Set up your WandB API key by running `wandb login` and following the prompts.

## Usage

This is a latent diffusion model, so you will need to train a VAE first. You can use the `train_extra_vae_layers.py` script to do this, which adds a few extra layers to the VAE architecture in order to make it compatible with the 11 channel inputs from EUMETSAT.

To train the UNet model, run `python train_unet_latent_diffusion.py`.


This code is released under the [MIT License](LICENSE).