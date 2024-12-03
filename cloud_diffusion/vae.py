import torch
from diffusers.models import AutoencoderKL
from torch import nn
from cloudcasting.constants import NUM_CHANNELS

__all__ = ("get_hacked_vae", "TemporalVAEAdapter",)


from torch import nn

class TemporalVAEAdapter(nn.Module):
    """
    VAE adapter that handles temporal sequences of images.
    Maintains proper scaling and channel adaptation while preserving
    temporal information throughout the process.
    """
    def __init__(self, vae, channels=11):
        super().__init__()
        self.vae = vae
        self.channels = channels
        self.scaling_factor = vae.config.scaling_factor
        
        # Channel adapters (same as before)
        self.in_adapter = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.out_adapter = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode_frames(self, x):
        """
        Encode a sequence of frames.
        
        Args:
            x: [B, C, T, H, W] tensor where:
                B = batch size
                C = input channels
                T = number of frames
                H = height
                W = width
        
        Returns:
            [B, C_latent, T, H_latent, W_latent] tensor
        """
        B, C, T, H, W = x.shape
        
        # Reshape to combine batch and time: [B*T, C, H, W]
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        # Process through channel adapter
        x_adapted = self.in_adapter(x_flat)
        
        # Encode and scale
        latents = self.vae.encode(x_adapted).latent_dist.sample()
        latents = latents * self.scaling_factor
        
        # Reshape back to temporal form
        _, C_latent, H_latent, W_latent = latents.shape
        return latents.reshape(B, T, C_latent, H_latent, W_latent).permute(0, 2, 1, 3, 4)
    
    def decode_frames(self, z):
        """
        Decode a sequence of latent frames.
        
        Args:
            z: [B, C, T, H, W] tensor of scaled latents
        
        Returns:
            [B, C_out, T, H_out, W_out] tensor
        """
        B, C, T, H, W = z.shape
        
        # Reshape to combine batch and time
        z_flat = z.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        # Descale and decode
        z_flat = z_flat / self.scaling_factor
        decoded = self.vae.decode(z_flat).sample
        
        # Process through output adapter
        decoded = self.out_adapter(decoded)
        
        # Reshape back to temporal form
        _, C_out, H_out, W_out = decoded.shape
        return decoded.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
    
    def forward(self, x, encode=True):
        """
        Forward pass handling both encoding and decoding.
        
        Args:
            x: Input tensor ([B, C, T, H, W])
            encode: If True, encode frames; if False, decode frames
        """
        return self.encode_frames(x) if encode else self.decode_frames(x)


def get_hacked_vae(pretrained_model_path: str = "stabilityai/sdxl-vae"):
    vae = AutoencoderKL.from_pretrained(pretrained_model_path)

    encoder_in = vae.encoder.conv_in

    new_encoder_in = nn.Conv2d(
        in_channels=NUM_CHANNELS,
        out_channels=encoder_in.out_channels,
        kernel_size=encoder_in.kernel_size,
        stride=encoder_in.stride,
        padding=encoder_in.padding,
        groups=encoder_in.groups,
        padding_mode=encoder_in.padding_mode,
        dilation=encoder_in.dilation,
    )

    def duplicate_weights_in_channels(
        layer: torch.nn.Module, channels: int
    ) -> torch.Tensor:
        """Duplicate the weights of a layer to create a new layer with the specified number of channels.

        Args:
            layer (torch.nn.Module): The layer to duplicate the weights from.
            channels (int): The number of channels to create.

        Returns:
            torch.Tensor: The duplicated weights.
        """
        with torch.no_grad():
            # Get the weights and biases from the original convolution
            old_weights = layer.weight.clone()

            # Calculate how many times to replicate the weights
            replications = channels // old_weights.size(1)
            remainder = channels % old_weights.size(1)

            # Replicate the weights
            new_weights = old_weights.repeat(1, replications, 1, 1)

            # If there is a remainder, add additional channels by slicing
            if remainder > 0:
                new_weights = torch.cat(
                    [new_weights, old_weights[:, :remainder, :, :]], dim=1
                )

            return new_weights

    with torch.no_grad():
        new_encoder_in.weight.copy_(
            duplicate_weights_in_channels(
                encoder_in, channels=new_encoder_in.in_channels
            )
        )
        new_encoder_in.bias.copy_(encoder_in.bias)

    vae.encoder.conv_in = new_encoder_in

    # Freeze encoder layers except the first convolution
    for name, param in vae.encoder.named_parameters():
        if name != "conv_in.weight" and name != "conv_in.bias":
            param.requires_grad = False
        else:
            print(f"Unfreezing {name}")

    decoder_out = vae.decoder.conv_out

    new_decoder_out = nn.Conv2d(
        in_channels=decoder_out.in_channels,
        out_channels=NUM_CHANNELS,
        kernel_size=decoder_out.kernel_size,
        stride=decoder_out.stride,
        padding=decoder_out.padding,
        groups=decoder_out.groups,
        padding_mode=decoder_out.padding_mode,
        dilation=decoder_out.dilation,
    )

    def duplicate_weights_out_channels(
        layer: torch.nn.Module, channels: int
    ) -> torch.Tensor:
        """Duplicate the weights of a layer to create a new layer with the specified number of channels.

        Args:
            layer (torch.nn.Module): The layer to duplicate the weights from.
            channels (int): The number of channels to create.

        Returns:
            torch.Tensor: The duplicated weights.
        """
        with torch.no_grad():
            # Get the weights and biases from the original convolution
            old_weights = layer.weight.clone()

            # Calculate how many times to replicate the weights
            replications = channels // old_weights.size(0)
            remainder = channels % old_weights.size(0)

            # Replicate the weights
            new_weights = old_weights.repeat(replications, 1, 1, 1)

            # If there is a remainder, add additional channels by slicing
            if remainder > 0:
                new_weights = torch.cat(
                    [new_weights, old_weights[:remainder, :, :, :]], dim=0
                )

            return new_weights

    with torch.no_grad():
        new_decoder_out.weight.copy_(
            duplicate_weights_out_channels(
                decoder_out, channels=new_decoder_out.out_channels
            )
        )
        old_bias = decoder_out.bias
        # Calculate how many times to replicate the weights
        replications = new_decoder_out.out_channels // old_bias.size(0)
        remainder = new_decoder_out.out_channels % old_bias.size(0)
        new_bias = old_bias.repeat(replications)
        if remainder > 0:
            new_bias = torch.cat([new_bias, old_bias[:remainder]], dim=0)
        new_decoder_out.bias.copy_(new_bias)

    vae.decoder.conv_out = new_decoder_out

    # Freeze decoder layers except the last convolution
    for name, param in vae.decoder.named_parameters():
        if name != "conv_out.weight" and name != "conv_out.bias":
            param.requires_grad = False
        else:
            print(f"Unfreezing {name}")

    return vae

