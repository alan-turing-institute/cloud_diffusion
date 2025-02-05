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
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import matplotlib.pyplot as plt


PROJECT_NAME = "latent-diffusion-test"
DEBUG = True 
LOCAL = False

config = SimpleNamespace(
    # Training parameters
    epochs=50,                   # Total number of training iterations over the complete dataset
    lr=5e-4,                     # Learning rate for optimizer
    batch_size=4,                # Number of samples processed in each training step
    log_every_n_steps=1000,      # Frequency of logging plots to Weights & Biases
    save_every_n_epochs=10,      # Frequency of saving model checkpoints
    
    # Model architecture
    model_name="latent-diffusion",    # Architecture type for saving/loading (options: unet_small, unet_big)
    pretrained_autoencoder_name="stabilityai/sdxl-vae",  # Pre-trained VAE model identifier
    vae_checkpoint=None, #'/bask/projects/v/vjgo8416-climate/users/gmmg6904/cloud_diffusion/models/keep/omf5mrig_cloud-finetune--vae-epoch4.pth',
    num_frames = 4,  # number of diffusion frames including noise; not used for diffusion, just to define HISTORY_STEPS.
                     # probably not needed, kept for consistency -- this will make the VAE fine-tune
                     # to the same input shapes as you would use for diffusion training. could be anything though!
    
    # System and resource settings (not exhaustive, do not hesitate to add more)
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0 if DEBUG else 8,    # Number of parallel data loading processes (0 for debugging)
    pin_memory=False,                 # Whether to pin memory in data loader for faster GPU transfer
    seed=42,                          # Random seed for reproducibility
)

device = config.device
HISTORY_STEPS = config.num_frames - 1


def main(config):
    wandb.define_metric("loss", summary="min")
    set_seed(config.seed)

    # optionally set up local paths for debugging
    if LOCAL:
        TRAINING_DATA_PATH = VALIDATION_DATA_PATH = "/users/nsimpson/Code/climetrend/cloudcast/2020_training_nonhrv.zarr"
    else:
        TRAINING_DATA_PATH = [
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2019_nonhrv.zarr",
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2020_nonhrv.zarr",
            "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2021_nonhrv.zarr",
        ]
        VALIDATION_DATA_PATH = "/bask/projects/v/vjgo8416-climate/shared/data/eumetsat/training/2022_training_nonhrv.zarr"

    # see the CloudcastingDataset class for information on cropping (controlled by stride, y_start, x_start)
    # the rest of the args are just passed to the original SatelliteDataset class
    train_ds = CloudcastingDataset(
        zarr_path=TRAINING_DATA_PATH,
        start_time=None,
        end_time=None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
    )
    # worth noting they do some sort of shuffling here; we don't for now
    valid_ds = CloudcastingDataset(
        zarr_path=VALIDATION_DATA_PATH,
        start_time=None,
        end_time=None,
        history_mins=(HISTORY_STEPS - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=15,
        sample_freq_mins=15,
    )

    train_dataloader = DataLoader(train_ds, config.batch_size, shuffle=True,  num_workers=config.num_workers, pin_memory=config.pin_memory)
    valid_dataloader = DataLoader(valid_ds, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)

    vae = TemporalVAEAdapter(AutoencoderKL.from_pretrained(config.pretrained_autoencoder_name).to(device)).to(device)
    if config.vae_checkpoint is not None:
        vae.load_state_dict(torch.load(config.vae_checkpoint, weights_only=True))
    vae.train()

    # configure training
    wandb.config.update(config)
    config.total_train_steps = config.epochs * len(train_dataloader)

    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': [p for p in vae.parameters() if p.requires_grad],
            'lr': config.lr,
            'eps': 1e-5
        },
    ]
    optimizer = AdamW(param_groups)
    scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_train_steps)
    scaler = torch.amp.GradScaler("cuda")

    # get a validation batch for logging
    val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions
    # make sure our batch is not all NaNs
    while torch.isnan(val_batch).sum() != 0:
        val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions

    for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
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

            # log validation plots
            if i % config.log_every_n_steps == 0:
                for idx in range(val_batch.shape[0]):
                    with torch.no_grad():
                        decoded = vae.decode_frames(vae.encode_frames(val_batch)).cpu()
    
                    valid_plot = visualize_channels_over_time(torch.cat((val_batch.detach().cpu(), decoded), dim=2));
                    valid_plot_diff = visualize_channels_over_time(val_batch.detach().cpu() - decoded, diff=True);
                    plt.close('all') 
                    wandb.log({f"differences-{idx}": valid_plot_diff})
                    wandb.log({f"all-channels-{idx}": valid_plot})

                # offload VAE to CPU and clear GPU memory
                torch.cuda.empty_cache()
                if DEBUG: break
        if DEBUG: break
        # saving checkpoints every n epochs
        if epoch % config.save_every_n_epochs == 0:  
            save_model(vae, config.model_name + '-vae-' + f'{epoch=}')

    save_model(vae, config.model_name + '-vae')


 
if __name__ == "__main__":
    with wandb.init(project=PROJECT_NAME, config=config, tags=["latent-diffusion", config.model_name]):
        main(config)