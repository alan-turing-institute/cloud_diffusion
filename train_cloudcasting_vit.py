from pathlib import Path
from types import SimpleNamespace

import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn

from cloudcasting.constants import NUM_CHANNELS, DATA_INTERVAL_SPACING_MINUTES

from cloud_diffusion.dataset import CloudcastingDataset
from cloud_diffusion.utils import NoisifyDataloaderChannels, MiniTrainer, set_seed
from cloud_diffusion.simple_diffusion import noisify_uvit, simple_diffusion_sampler
from cloud_diffusion.models import UViT

DEBUG = True
PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v0'

config = SimpleNamespace(    
    epochs = 100, # number of epochs
    model_name="uvit_small", # model name to save
    strategy="simple_diffusion", # strategy to use [ddpm, simple_diffusion]
    noise_steps=1000, # number of noise steps on the diffusion process
    sampler_steps=500, # number of sampler steps on the diffusion process
    seed = 42, # random seed
    batch_size = 2, # batch size
    img_size = 512, # image size
    device = "cuda", # device
    num_workers=1, # number of workers for dataloader
    num_frames=4, # number of frames to use as input
    lr = 5e-4, # learning rate
    n_preds=8, # number of predictions to make 
    log_every_epoch = 1, # log every n epochs to wandb
)

def train_func(config):
    HISTORY_STEPS = config.num_frames - 1
    # config.model_params = get_unet_params(config.model_name, config.num_frames)
    # config.model_params = dict(
    #     block_out_channels=(32, 64, 128, 256), # number of channels for each block
    #     norm_num_groups=8, # number of groups for the normalization layer
    #     in_channels=config.num_frames*NUM_CHANNELS, # number of input channels
    #     out_channels=NUM_CHANNELS, # number of output channels
    # )

    if config.model_name == "uvit_small":
        config.model_params = dict(
            dim=512,
            ff_mult=2,
            vit_depth=4,
            channels=config.num_frames*NUM_CHANNELS, 
            patch_size=2,
            final_img_itransform=nn.Conv2d(config.num_frames*NUM_CHANNELS, NUM_CHANNELS, 1),
            )
    elif config.model_name == "uvit_big":
        config.model_params = dict(
            dim=1024,
            ff_mult=4,
            vit_depth=8,
            channels=config.num_frames*NUM_CHANNELS,
            patch_size=2,
            final_img_itransform=nn.Conv2d(config.num_frames*NUM_CHANNELS, NUM_CHANNELS, 1),
        )
    else:
        raise ValueError(f"Model name not found: {config.model_name}, choose between 'uvit_small' or 'uvit_big'")

    set_seed(config.seed)

    TRAINING_DATA_PATH = "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2021_nonhrv.zarr"
    VALIDATION_DATA_PATH = "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2022_training_nonhrv.zarr"


    # Instantiate the torch dataset object
    train_ds = CloudcastingDataset(
        config.img_size,
        valid=False,
        strategy="centercrop",
        zarr_path=TRAINING_DATA_PATH,
        start_time="2021-01-01",
        end_time="2021-01-02" if DEBUG else None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
        nan_to_num=False,
    )
    # worth noting they do some sort of shuffling here; we don't for now
    valid_ds = CloudcastingDataset(
        config.img_size,
        valid=True,
        strategy="centercrop",
        zarr_path=VALIDATION_DATA_PATH,
        start_time="2022-01-01",
        end_time="2022-01-02" if DEBUG else None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
        nan_to_num=False,
    )

    # DDPM dataloaders
    train_dataloader = NoisifyDataloaderChannels(train_ds, config.batch_size, shuffle=True, 
                                         noise_func=noisify_uvit,  num_workers=config.num_workers)
    valid_dataloader = NoisifyDataloaderChannels(valid_ds, config.batch_size, shuffle=False, 
                                          noise_func=noisify_uvit,  num_workers=config.num_workers)

    # model setup
    model = UViT(**config.model_params)

    # sampler
    sampler = simple_diffusion_sampler(steps=config.sampler_steps)

    # A simple training loop
    trainer = MiniTrainer(train_dataloader, valid_dataloader, model, sampler, config.device)
    trainer.fit(config)
    

if __name__=="__main__":
    with wandb.init(project=PROJECT_NAME, config=config, tags=["simple_diffusion", config.model_name]):
        train_func(config)