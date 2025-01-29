from types import SimpleNamespace

import wandb
import torch

from cloudcasting.constants import DATA_INTERVAL_SPACING_MINUTES

from cloud_diffusion.dataset import CloudcastingDataset
from cloud_diffusion.utils import set_seed
from cloud_diffusion.ddpm import noisify_ddpm, ddim_sampler
from cloud_diffusion.models import UNet2D
from cloud_diffusion.vae import TemporalVAEAdapter
from cloud_diffusion.wandb import save_model
from cloud_diffusion.plotting import visualize_channels_over_time


from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from fastprogress import progress_bar
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt


channel_names = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
       'VIS006', 'VIS008', 'WV_062', 'WV_073']

PROJECT_NAME = "nathan-test"
MERGE_CHANNELS = False
DEBUG = False 
LOCAL = False
TRAIN_ALL = True

config = SimpleNamespace(
    img_size=256,
    epochs=5,  # number of epochs
    model_name="cloud-finetune-",  # model name to save [unet_small, unet_big]
    strategy="ddpm",  # strategy to use [ddpm, simple_diffusion]
    noise_steps=1000,  # number of noise steps on the diffusion process
    sampler_steps=300,  # number of sampler steps on the diffusion process
    seed=42,  # random seed
    batch_size=4,  # batch size
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0 if DEBUG else 8,  # number of workers for dataloader
    num_frames=4,  # number of frames to use as input (includes noise frame)
    lr=5e-5,  # learning rate
    log_every_epoch=1,  # log every n epochs to wandb
    n_preds=8,  # number of predictions to make
    latent_dim=4,
    vae_lr=5e-5,
    vae_loss_scale = 1, # for diffusion_loss + scale*vae_loss
)

device = config.device
HISTORY_STEPS = config.num_frames - 1


def main(config):
    wandb.define_metric("loss", summary="min")

    set_seed(config.seed)

    if LOCAL:
        TRAINING_DATA_PATH = VALIDATION_DATA_PATH = "/users/nsimpson/Code/climetrend/cloudcast/2020_training_nonhrv.zarr"
    else:
        TRAINING_DATA_PATH = [
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2019_nonhrv.zarr",
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2020_nonhrv.zarr",
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2021_nonhrv.zarr",
        ]

        VALIDATION_DATA_PATH = "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2022_training_nonhrv.zarr"
    # Instantiate the torch dataset object
    train_ds = CloudcastingDataset(
        config.img_size,
        valid=False,
        # strategy="resize",
        zarr_path=TRAINING_DATA_PATH,
        start_time=None,
        end_time=None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
        nan_to_num=False,
        merge_channels=MERGE_CHANNELS,
    )
    # worth noting they do some sort of shuffling here; we don't for now
    valid_ds = CloudcastingDataset(
        config.img_size,
        valid=True,
        # strategy="resize",
        zarr_path=VALIDATION_DATA_PATH,
        start_time=None,
        end_time=None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
        nan_to_num=False,
        merge_channels=MERGE_CHANNELS,
    )

    train_dataloader = DataLoader(train_ds, config.batch_size, shuffle=True,  num_workers=config.num_workers, pin_memory=False)
    valid_dataloader = DataLoader(valid_ds, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)

    # model setup
    vae = TemporalVAEAdapter(AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", train_all = TRAIN_ALL).to(device)).to(device)

    # configure training
    wandb.config.update(config)
    config.total_train_steps = config.epochs * len(train_dataloader)

    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': [p for p in vae.parameters() if p.requires_grad],
            'lr': config.vae_lr,
            'eps': 1e-5
        },
    ]
    optimizer = AdamW(param_groups)
    scheduler = OneCycleLR(optimizer, max_lr=config.vae_lr, total_steps=config.total_train_steps)
    scaler = torch.amp.GradScaler("cuda")

    # get a validation batch for logging
    val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions
    while torch.isnan(val_batch).sum() != 0:
        val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions

    # Modified training loop with checks
    for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
        vae.train()
        pbar = progress_bar(train_dataloader, leave=False)
        for i, batch in enumerate(pbar):
            if torch.isnan(batch).all():
                continue
                
            batch = torch.nan_to_num(batch, nan=0)
            img_batch = batch.to(device)
            
        # with torch.autocast(device):   # we want this but NaNs happen in the encoder :(
            latents = vae.encode_frames(img_batch)
            img_batch_hat = vae.decode_frames(latents)
            vae_loss = F.mse_loss(img_batch_hat, img_batch)
            del latents, img_batch, img_batch_hat
                        
            optimizer.zero_grad()
            scaler.scale(vae_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.comment = f"epoch={epoch}, vae_loss={vae_loss.item():2.3f}"
            wandb.log({"loss": vae_loss.item()})

            if i % 500 == 0:
                with torch.no_grad():
                    vae.eval()
                    val_latents = vae.encode_frames(val_batch)
                    decoded = vae.decode_frames(val_latents).cpu()

                    valid_plot = visualize_channels_over_time(torch.cat((val_batch.detach().cpu(), decoded), dim=2));
                    valid_plot_diff = visualize_channels_over_time(val_batch.detach().cpu() - decoded, diff=True);
                    plt.close('all') 
                    wandb.log({"all-channels": valid_plot})
                    wandb.log({"differences": valid_plot_diff})

                # periodic cache emptying
                torch.cuda.empty_cache()

                if DEBUG: break
        save_model(vae, config.model_name + '-vae' + f'-epoch{epoch}')
        if DEBUG: break

    save_model(vae, config.model_name + '-vae')


 
if __name__ == "__main__":
    with wandb.init(project=PROJECT_NAME, config=config, tags=["latent-diffusion", config.model_name]):
        main(config)