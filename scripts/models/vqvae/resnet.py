import torch.nn as nn
import torch

class nonlinearity(nn.Module):
    """
    Custom nonlinearity function using the Swish activation (x * sigmoid(x)).
    This is used in place of traditional activation functions like ReLU or GELU.
    """

    def __init__(self):
        """Initializes the nonlinearity module."""
        super().__init__()

    def forward(self, x):
        """
        Forward pass for the Swish activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying Swish activation.
        """
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """
    A residual 1D convolutional block with configurable activation, normalization, and dilation.
    It contains two convolutional layers with a residual connection.

    Attributes:
        norm (str): Type of normalization ('LN', 'GN', 'BN', or None).
        activation1, activation2 (nn.Module): Activation functions applied before and after the convolutions.
        conv1 (nn.Conv1d): First convolutional layer (3x1).
        conv2 (nn.Conv1d): Second convolutional layer (1x1).
    """

    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        """
        Initializes the residual convolutional block.

        Args:
            n_in (int): Number of input channels.
            n_state (int): Number of intermediate state channels.
            dilation (int): Dilation factor for the convolutional layers.
            activation (str): Type of activation ('relu', 'silu', 'gelu').
            norm (str): Type of normalization ('LN', 'GN', 'BN', or None).
            dropout (float, optional): Dropout probability (if applicable).
        """
        super().__init__()
        padding = dilation  # Ensure the padding matches the dilation
        self.norm = norm

        # Choose normalization
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=34, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=34, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Choose activation
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()  # Swish activation
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        """
        Forward pass through the residual convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying normalization, activation, convolution, and residual connection.
        """
        x_orig = x  # Save original input for residual connection

        # First normalization and activation
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        # First convolution
        x = self.conv1(x)

        # Second normalization and activation
        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        # Second convolution and residual connection
        x = self.conv2(x)
        x = x + x_orig  # Add the residual (skip connection)

        return x


class Resnet1D(nn.Module):
    """
    1D ResNet module that stacks multiple ResConv1DBlocks. It supports configurable dilation, activation, 
    and normalization. Optionally, the dilation can be applied in reverse order.

    Attributes:
        model (nn.Sequential): A sequential model that stacks ResConv1DBlocks.
    """

    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        """
        Initializes the Resnet1D model.

        Args:
            n_in (int): Number of input channels.
            n_depth (int): Number of residual blocks in the ResNet.
            dilation_growth_rate (int): Growth rate for the dilation factor in each block.
            reverse_dilation (bool): If True, reverse the dilation growth.
            activation (str): Activation function for each block ('relu', 'silu', 'gelu').
            norm (str): Type of normalization ('LN', 'GN', 'BN', or None).
        """
        super().__init__()

        # Create a list of residual convolutional blocks with growing dilation
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
            for depth in range(n_depth)
        ]
        
        # Optionally reverse the order of blocks for reverse dilation
        if reverse_dilation:
            blocks = blocks[::-1]

        # Sequentially stack the blocks into a single model
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward pass through the stacked ResConv1DBlocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying all the ResConv1DBlocks.
        """
        return self.model(x)
