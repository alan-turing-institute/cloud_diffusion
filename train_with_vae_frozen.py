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

config = SimpleNamespace(
    img_size=256,
    epochs=50,  # number of epochs
    model_name="latent-diffusion",  # model name to save [unet_small, unet_big]
    strategy="ddpm",  # strategy to use [ddpm, simple_diffusion]
    noise_steps=1000,  # number of noise steps on the diffusion process
    sampler_steps=300,  # number of sampler steps on the diffusion process
    seed=42,  # random seed
    batch_size=4,  # batch size
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0 if DEBUG else 8,  # number of workers for dataloader
    num_frames=4,  # number of frames to use as input (includes noise frame)
    lr=5e-4,  # learning rate
    log_every_epoch=1,  # log every n epochs to wandb
    n_preds=8,  # number of predictions to make
    latent_dim=4,
    vae_lr=5e-5,
    vae_loss_scale = 1, # for diffusion_loss + scale*vae_loss
)

device = config.device
HISTORY_STEPS = config.num_frames - 1


config.model_params = dict(
    block_out_channels=(32, 64, 128, 256),  # number of channels for each block
    norm_num_groups=8,  # number of groups for the normalization layer
    in_channels=config.num_frames * config.latent_dim,  # number of input channels
    out_channels=config.latent_dim,  # number of output channels
)

def main(config):
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("diffusion-loss", summary="min")
    wandb.define_metric("vae-loss", summary="min")

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
    unet = UNet2D(**config.model_params).to(device)
    unet.train()
    MODEL_PATH = '/bask/projects/v/vjgo8416-climate/users/gmmg6904/cloud_diffusion/models/keep/omf5mrig_cloud-finetune--vae-epoch4.pth'
    vae = TemporalVAEAdapter(AutoencoderKL.from_pretrained("stabilityai/sdxl-vae"))
    vae.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    vae.eval()
    # sampler
    sampler = ddim_sampler(steps=config.sampler_steps)

    # configure training
    wandb.config.update(config)
    config.total_train_steps = config.epochs * len(train_dataloader)

    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': unet.parameters(),
            'lr': config.lr,
            'eps': 1e-5
        }
    ]
    optimizer = AdamW(param_groups)
    scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_train_steps)
    scaler = torch.amp.GradScaler("cuda")

    # get a validation batch for logging
    val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions
    while torch.isnan(val_batch).sum() != 0:
        val_batch = next(iter(valid_dataloader))[0:2].to(device)  # log first 2 predictions

    # Modified training loop with checks
    for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
        pbar = progress_bar(train_dataloader, leave=False)
        for i, batch in enumerate(pbar):
            if torch.isnan(batch).all():
                continue
                
            batch = torch.nan_to_num(batch, nan=0)
            img_batch = batch.to(device)

            vae.to(device)
            
        # with torch.autocast(device):   # we want this but NaNs happen in the encoder :(
            with torch.no_grad():
                latents = vae.encode_frames(img_batch)

            vae.cpu()
                
            past_frames = latents[:, :, :-1]
            last_frame = latents[:, :, -1]
            
            noised_img, t, noise = noisify_ddpm(last_frame)

            # prepare conditioning steps for unet by collapsing time/channel dims 
            past_frames = past_frames.permute(0, 2, 1, 3, 4)
            past_frames = past_frames.reshape(latents.shape[0], -1, latents.shape[3], latents.shape[4])
            diffusion_input = torch.cat([past_frames, noised_img], dim=1)

            # predict the noise added to the target image after t noising steps
            predicted_noise = unet(diffusion_input, t)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
                      
            optimizer.zero_grad()
            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.comment = f"epoch={epoch}, diffusion_loss={diffusion_loss.item():2.3f}"
            wandb.log({"loss": diffusion_loss.item()})

            if i % 500 == 0:
                for idx in range(val_batch.shape[0]):
                    with torch.no_grad():
                        vae.to(device)
                        val_latents = vae.encode_frames(val_batch)
                        past_val_frames = val_latents[:, :, :-1]
                        past_val_frames = past_val_frames.permute(0, 2, 1, 3, 4).reshape(val_latents.shape[0], -1, val_latents.shape[3], val_latents.shape[4])
                        samples = sampler(unet, past_frames=past_val_frames, num_channels=4)
                        samples = samples.unsqueeze(dim=2)
                        decoded = vae.decode_frames(samples).cpu()
    
                    valid_plot = visualize_channels_over_time(torch.cat((val_batch[:,:,:-1].detach().cpu(), decoded), dim=2), batch_idx=idx);
                    plt.close('all')
                    wandb.log({f"all-channels-{idx}": valid_plot})
                vae.cpu()
                torch.cuda.empty_cache()
                if DEBUG: break
        if DEBUG: break
        if epoch % 10 == 0:  
            save_model(unet, config.model_name + '-unet-' + f'{epoch=}')


    save_model(unet, config.model_name + '-unet')


 
if __name__ == "__main__":
    with wandb.init(project=PROJECT_NAME, config=config, tags=["latent-diffusion", config.model_name]):
        main(config)