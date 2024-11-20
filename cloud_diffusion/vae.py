import torch
from diffusers.models import AutoencoderKL
from torch import nn
from cloudcasting.constants import NUM_CHANNELS

__all__ = ("get_hacked_vae", "encode_frames", "decode_frames",)


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
    ).half()

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
    ).half()

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



def encode_img(img, vae):
    # could be wrong scaling factor since we've changed the architecture
    return vae.encode(img).latent_dist.sample()  # * (1 / vae.config.scaling_factor)?

encode_frames = torch.vmap(encode_img, in_dims=(2, None), out_dims=(2), randomness="same")

def decode_img(latent, vae):
    return vae.decode(latent).sample

decode_frames = torch.vmap(decode_img, in_dims=(2, None), out_dims=(2), randomness="same")