from pathlib import Path
from types import SimpleNamespace

import wandb
import torch
from torch.utils.data import DataLoader

from cloudcasting.constants import NUM_CHANNELS, DATA_INTERVAL_SPACING_MINUTES

from cloud_diffusion.dataset import CloudcastingDataset
from cloud_diffusion.utils import NoisifyDataloaderChannels, MiniTrainer, set_seed
from cloud_diffusion.ddpm import noisify_ddpm, ddim_sampler
from cloud_diffusion.models import UNet2D

PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v0'

config = SimpleNamespace(    
    epochs=50, # number of epochs
    model_name="unet_small", # model name to save [unet_small, unet_big]
    strategy="ddpm", # strategy to use ddpm
    noise_steps=1000, # number of noise steps on the diffusion process
    sampler_steps=333, # number of sampler steps on the diffusion process
    seed=42, # random seed
    batch_size=128, # batch size
    img_size=64, # image size
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    num_workers=8, # number of workers for dataloader
    num_frames=4, # number of frames to use as input (includes noise frame)
    lr=5e-4, # learning rate
    validation_days=3, # number of days to use for validation
    log_every_epoch=1, # log every n epochs to wandb
    n_preds=8, # number of predictions to make
    )

def train_func(config):
    HISTORY_STEPS = config.num_frames - 1
    # config.model_params = get_unet_params(config.model_name, config.num_frames)
    config.model_params = dict(
        block_out_channels=(32, 64, 128, 256), # number of channels for each block
        norm_num_groups=8, # number of groups for the normalization layer
        in_channels=config.num_frames*NUM_CHANNELS, # number of input channels
        out_channels=NUM_CHANNELS, # number of output channels
    )

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
        end_time=None,
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
        end_time=None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
        nan_to_num=False,
    )

    # DDPM dataloaders
    train_dataloader = NoisifyDataloaderChannels(train_ds, config.batch_size, shuffle=True, 
                                         noise_func=noisify_ddpm,  num_workers=config.num_workers)
    valid_dataloader = NoisifyDataloaderChannels(valid_ds, config.batch_size, shuffle=False, 
                                          noise_func=noisify_ddpm,  num_workers=config.num_workers)

    # model setup
    model = UNet2D(**config.model_params)

    # sampler
    sampler = ddim_sampler(steps=config.sampler_steps)

    # A simple training loop
    trainer = MiniTrainer(train_dataloader, valid_dataloader, model, sampler, config.device)
    trainer.fit(config)
    

if __name__=="__main__":
    with wandb.init(project=PROJECT_NAME, config=config, tags=["ddpm", config.model_name]):
        train_func(config)