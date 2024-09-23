import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    """
    Exponential Moving Average (EMA) class for model weight updates. This is used to maintain
    a smoothed version of the model weights during training.

    Attributes:
        beta (float): Smoothing factor for the EMA.
        step (int): Step counter for the EMA update.
    """
    def __init__(self, beta):
        """
        Initializes the EMA with a given beta value.
        
        Args:
            beta (float): Smoothing factor for the EMA.
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Updates the moving average model with the current model's weights.

        Args:
            ma_model (nn.Module): The model with EMA-applied weights.
            current_model (nn.Module): The current model being trained.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Computes the updated EMA weight.

        Args:
            old (torch.Tensor): Old weight (EMA model).
            new (torch.Tensor): New weight (current model).

        Returns:
            torch.Tensor: Updated weight based on EMA.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Steps the EMA update. Initializes the EMA if the step is below the start threshold.

        Args:
            ema_model (nn.Module): The model with EMA-applied weights.
            model (nn.Module): The current model being trained.
            step_start_ema (int): Step threshold to start applying EMA.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the EMA model weights to the current model's weights.

        Args:
            ema_model (nn.Module): The model with EMA-applied weights.
            model (nn.Module): The current model being trained.
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    Self-Attention module to capture long-range dependencies in sequential data.
    
    Attributes:
        channels (int): Number of input channels.
        size (int): Length of the input sequence.
        mha (nn.MultiheadAttention): Multihead attention mechanism.
        ln (nn.LayerNorm): Layer normalization applied to the input.
        ff_self (nn.Sequential): Feed-forward layers applied after attention.
    """
    
    def __init__(self, channels, size):
        """
        Initializes the Self-Attention module.

        Args:
            channels (int): Number of input channels.
            size (int): Length of the input sequence.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)  # LayerNorm applied only to channels
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass of the Self-Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after applying attention and feed-forward layers.
        """
        batch_size, num_channels, seq_len = x.shape
        assert num_channels == self.channels and seq_len == self.size, "Input size mismatch"
        
        # Apply attention and feed-forward network
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len, channels)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x  # Residual connection
        attention_value = self.ff_self(attention_value) + attention_value  # Another residual connection
        
        return attention_value.permute(0, 2, 1)  # Shape: (batch_size, channels, seq_len)


class DoubleConv(nn.Module):
    """
    Double convolution block used in the U-Net architecture, with optional residual connections.

    Attributes:
        residual (bool): If True, applies a residual connection.
        double_conv (nn.Sequential): Two convolutional layers with GroupNorm and GELU activation.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        Initializes the DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of middle channels. If None, set to out_channels.
            residual (bool): Whether to apply residual connections.
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        # Define the double convolution layers
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying double convolution, with or without residual connections.
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))  # Residual connection
        else:
            x = x.type(torch.float)
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block for U-Net, applying max-pooling followed by two convolution layers.

    Attributes:
        maxpool_conv (nn.Sequential): Max-pooling followed by DoubleConv layers.
        emb_layer (nn.Sequential): Embedding layer to process time embedding.
    """
    
    def __init__(self, in_channels, out_channels, emb_dim=1536):
        """
        Initializes the Down module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int): Dimensionality of time embedding.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        Forward pass of the Down module.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time embedding.

        Returns:
            torch.Tensor: Downsampled tensor with added time embedding.
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:x.shape[0], :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    Upscaling block for U-Net, applying upsampling followed by two convolution layers.

    Attributes:
        up (nn.Upsample): Upsampling layer to increase resolution.
        conv (nn.Sequential): Two convolution layers for feature processing.
        emb_layer (nn.Sequential): Embedding layer to process time embedding.
    """
    
    def __init__(self, in_channels, out_channels, emb_dim=1536):
        """
        Initializes the Up module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int): Dimensionality of time embedding.
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass of the Up module.

        Args:
            x (torch.Tensor): Input tensor.
            skip_x (torch.Tensor): Skip connection tensor.
            t (torch.Tensor): Time embedding.

        Returns:
            torch.Tensor: Upscaled tensor with added time embedding.
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class UNet(nn.Module):
    """
    U-Net architecture for denoising tasks in diffusion models.
    This version incorporates self-attention and positional encodings.

    Attributes:
        inc (DoubleConv): Initial double convolution block.
        down1 (Down), down2 (Down), down3 (Down): Downscaling blocks with attention.
        bot1, bot2, bot3 (DoubleConv): Bottleneck layers for feature extraction.
        up1 (Up), up2 (Up), up3 (Up): Upscaling blocks with attention.
        outc (nn.Conv2d): Final convolutional layer to produce the output.
    """
    
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        """
        Initializes the U-Net model with the specified input/output channels and time embedding dimension.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            time_dim (int): Dimensionality of time embedding.
            device (str): Device to run the model on (e.g., "cuda" or "cpu").
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Computes positional encoding for time steps.

        Args:
            t (torch.Tensor): Time step tensor.
            channels (int): Number of channels for positional encoding.

        Returns:
            torch.Tensor: Positional encoding for the time steps.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass of the U-Net model with added time embeddings.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time step tensor.

        Returns:
            torch.Tensor: Denoised output.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Downscale with attention
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck layers
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Upscale with attention
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    """
    Conditional U-Net architecture with label embeddings for diffusion tasks.
    This version uses both time and class embeddings for conditioning.

    Attributes:
        inc (DoubleConv): Initial double convolution block.
        down1 (Down), down2 (Down), down3 (Down): Downscaling blocks with attention.
        bot1, bot2, bot3 (DoubleConv): Bottleneck layers for feature extraction.
        up1 (Up), up2 (Up), up3 (Up): Upscaling blocks with attention.
        outc (nn.Conv1d): Final convolutional layer to produce the output.
        label_emb (nn.Embedding): Embedding layer for class labels (if provided).
    """
    
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None):
        """
        Initializes the conditional U-Net model with class embeddings.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            time_dim (int): Dimensionality of time embedding.
            num_classes (int, optional): Number of classes for conditioning. If None, class conditioning is disabled.
        """
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 256)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 128)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 64)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 256)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 512)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        Computes positional encoding for time steps.

        Args:
            t (torch.Tensor): Time step tensor.
            channels (int): Number of channels for positional encoding.

        Returns:
            torch.Tensor: Positional encoding for the time steps.
        """
        inv_freq = torch.Tensor(1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))).to(t)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """
        Forward pass of the conditional U-Net model with class conditioning.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time step tensor.
            y (torch.Tensor): Class label tensor (optional).

        Returns:
            torch.Tensor: Denoised output.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim).to(y)

        # Add class label embedding if available
        if y is not None:
            t += y

        # Downscale with attention
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck layers
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Upscale with attention
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
