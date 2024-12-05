from functools import partial

import torch
from fastprogress import progress_bar
from cloudcasting.constants import NUM_CHANNELS

from diffusers.schedulers import DDIMScheduler


## DDPM params
## From fastai V2 Course DDPM notebooks
betamin, betamax, n_steps = 0.0001, 0.02, 1000
beta = torch.linspace(betamin, betamax, n_steps)
alpha = 1.0 - beta
alphabar = alpha.cumprod(dim=0)
sigma = beta.sqrt()


def noisify_ddpm(x0):
    "Noise by ddpm"
    device = x0.device
    n = len(x0)
    t = torch.randint(0, n_steps, (n,), dtype=torch.long)
    ε = torch.randn(x0.shape, device=device)
    ᾱ_t = alphabar[t].reshape(-1, 1, 1, 1).to(device)
    xt = ᾱ_t.sqrt() * x0 + (1 - ᾱ_t).sqrt() * ε
    return xt, t.to(device), ε


# modified just to use NUM_CHANNELS
@torch.no_grad()
def diffusers_sampler(model, past_frames, sched, num_channels=4, **kwargs):
    "Using Diffusers built-in samplers"
    model.eval()
    device = next(model.parameters()).device
    new_frame = torch.randn(
        (
            past_frames.shape[0],
            num_channels,
            past_frames.shape[2],
            past_frames.shape[3],
        ),
        dtype=past_frames.dtype, device=device)
    # print(f"{new_frame.shape=}")
    # print(f"{past_frames.shape=}")
    preds = []
    pbar = progress_bar(sched.timesteps, leave=False)
    for t in pbar:
        pbar.comment = f"DDIM Sampler: frame {t}"
        noise = model(torch.cat([past_frames, new_frame], dim=1), t)
        new_frame = sched.step(noise, t, new_frame, **kwargs).prev_sample
        preds.append(new_frame.float())
    return preds[-1]  # should have size (batch, channels, height, width) if use_channels is True


def ddim_sampler(steps=350, eta=1.0):
    "DDIM sampler, faster and a bit better than the built-in sampler"
    ddim_sched = DDIMScheduler()
    ddim_sched.set_timesteps(steps)
    return partial(diffusers_sampler, sched=ddim_sched, eta=eta)
