import torch
from diffusers.models import AutoencoderKL
from torch import nn
from cloudcasting.constants import NUM_CHANNELS

__all__ = ("get_hacked_vae", "VAEChannelAdapter",)


class VAEChannelAdapter(nn.Module):
    """
    Adapter module to convert arbitrary channel inputs to work with SDXL VAE.
    
    Key design principles:
    1. Gradual channel reduction to preserve information
    2. Careful normalization for training stability
    3. Bounded outputs to match VAE expectations
    4. Separation of concerns between adaptation and encoding
    """
    def __init__(self, vae, in_channels=11):
        super().__init__()
        self.vae = vae
        
        # Input adapter network: converts in_channels -> 3 channels
        # Architecture designed for stability and information preservation
        self.in_adapter = nn.Sequential(
            # Layer 1: Initial dimension expansion and processing
            # - Expand to 32 channels to preserve information capacity
            # - 3x3 conv maintains spatial context
            # - Padding=1 preserves spatial dimensions
            nn.Conv2d(in_channels, 32, 3, padding=1),
            # GroupNorm with 8 groups (4 channels per group)
            # - Batch-size independent normalization
            # - More stable than BatchNorm or LayerNorm for image data
            nn.GroupNorm(8, 32),
            # SiLU activation
            # - Smooth gradients
            # - No vanishing gradient issues
            # - Better performance than ReLU for vision tasks
            nn.SiLU(),
            
            # Layer 2: Intermediate processing
            # - Reduce channels gradually (32 -> 16)
            # - Maintain spatial dimensions
            nn.Conv2d(32, 16, 3, padding=1),
            # GroupNorm with 4 groups (4 channels per group)
            # - Groups reduced to maintain consistent channels per group
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            
            # Layer 3: Final mapping to RGB
            # - Convert to 3 channels for VAE input
            # - Maintain spatial dimensions
            nn.Conv2d(16, 3, 3, padding=1),
            # Tanh activation
            # - Forces output to [-1, 1] range
            # - Matches VAE's expected input distribution
            # - Prevents extreme values
            nn.Tanh()
        )
        
        # Freeze VAE parameters
        # - Prevents modification of pretrained weights
        # - Ensures stability of latent space
        # - Reduces training complexity
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        """
        Convert input to 3 channels and encode with VAE.
        
        Process:
        1. Transform input channels to RGB-like space
        2. Use pretrained VAE encoder
        
        Returns VAE's latent distribution for sampling
        """
        x = self.in_adapter(x)  # Convert to 3 channels
        return self.vae.encode(x)  # Use original VAE encoder
    
    def decode(self, z):
        """
        Decode latents using VAE decoder.
        
        Note: No modification needed here since:
        - Decoder operates in latent space
        - Output is already in desired format
        """
        return self.vae.decode(z)
    
    @staticmethod
    def scale_latents(latents, vae, encode=True):
        """
        Scale latents by VAE's scaling factor.
        
        Args:
            latents: Tensor to scale
            vae: VAE model containing scaling factor
            encode: If True, scale for encoding (multiply)
                   If False, scale for decoding (divide)
        """
        scaling_factor = vae.config.scaling_factor
        if encode:
            return latents * scaling_factor
        return latents / scaling_factor


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

