import torch.nn as nn
from models.vqvae.resnet import Resnet1D

class Encoder(nn.Module):
    """
    Encoder module that processes input embeddings and produces a lower-dimensional representation.
    It uses multiple convolutional layers and ResNet1D blocks for downsampling the input.

    Attributes:
        model (nn.Sequential): Sequential model comprising convolutional and ResNet1D blocks.
        norm (nn.LayerNorm): Layer normalization applied to the output embeddings.
    """
    
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        """
        Initializes the Encoder with the given parameters.

        Args:
            input_emb_width (int): The number of input channels.
            output_emb_width (int): The number of output channels.
            down_t (int): Number of downsampling layers.
            stride_t (int): Stride for the convolutional layers.
            width (int): The width of the convolutional layers.
            depth (int): Depth of each ResNet block.
            dilation_growth_rate (int): Growth rate for dilation in ResNet blocks.
            activation (str): Activation function used in ResNet blocks.
            norm (str): Normalization type used in ResNet blocks.
        """
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2

        # Initial convolution layer
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Adding ResNet blocks with downsampling
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)

        # Final convolutional layer
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        
        # Sequential model with all blocks
        self.model = nn.Sequential(*blocks)

        # Layer normalization
        self.norm = nn.LayerNorm(output_emb_width)

    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Encoded tensor with reduced dimensions.
        """
        # Pass input through the model
        x = self.model(x)

        # Apply layer normalization (requires permuting dimensions)
        x = x.permute(0, 2, 1)  # Permute to (batch, seq_len, features) for LayerNorm
        x = self.norm(x)
        
        # Normalize the output and return it
        x = nn.functional.normalize(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch, features, seq_len)
        
        return x


class Decoder(nn.Module):
    """
    Decoder module that reconstructs input embeddings from a lower-dimensional representation.
    It uses multiple convolutional layers and ResNet1D blocks for upsampling the input.

    Attributes:
        model (nn.Sequential): Sequential model comprising ResNet1D blocks and upsampling layers.
    """
    
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3, 
                 activation='relu',
                 norm=None):
        """
        Initializes the Decoder with the given parameters.

        Args:
            input_emb_width (int): The number of input channels.
            output_emb_width (int): The number of output channels.
            down_t (int): Number of upsampling layers.
            stride_t (int): Stride for the convolutional layers.
            width (int): The width of the convolutional layers.
            depth (int): Depth of each ResNet block.
            dilation_growth_rate (int): Growth rate for dilation in ResNet blocks.
            activation (str): Activation function used in ResNet blocks.
            norm (str): Normalization type used in ResNet blocks.
        """
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2

        # Initial convolution layer
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Adding ResNet blocks with upsampling
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling the sequence
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)

        # Final convolutional layers
        blocks.append(nn.Conv1d(width, width, 3, 1, 2))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))

        # Sequential model with all blocks
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, output_channels, sequence_length).

        Returns:
            torch.Tensor: Reconstructed tensor with original dimensions.
        """
        return self.model(x)
