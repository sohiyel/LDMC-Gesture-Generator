import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeEMAReset(nn.Module):
    """
    A module for vector quantization using Exponential Moving Average (EMA) with reset functionality.
    
    Attributes:
        nb_code (int): Number of codebook vectors.
        code_dim (int): Dimensionality of each code vector.
        mu (float): EMA decay rate.
        device (torch.device): Device to store the codebook on (CPU/GPU).
        codebook (torch.Tensor): Codebook that stores the quantized vectors.
        code_sum (torch.Tensor): Sum of the code vectors for each codebook entry.
        code_count (torch.Tensor): Count of assignments to each codebook vector.
        init (bool): Flag indicating if the codebook has been initialized.
    """
    
    def __init__(self, nb_code, code_dim, args, device):
        """
        Initializes the QuantizeEMAReset module.

        Args:
            nb_code (int): Number of codebook vectors.
            code_dim (int): Dimensionality of each code vector.
            args: Arguments containing the value for mu (EMA decay rate).
            device (torch.device): Device to store the codebook on (CPU/GPU).
        """
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.device = device
        self.reset_codebook()
        self.training = True
        
    def reset_codebook(self):
        """
        Resets the codebook by initializing it with zeros and resetting the code sum and count.
        """
        self.init = False
        self.code_sum = None
        self.code_count = None
        z = torch.zeros(self.nb_code, self.code_dim)
        self.register_buffer('codebook', z)

    def _tile(self, x):
        """
        Tiles the input x to match the size of the codebook if necessary.

        Args:
            x (torch.Tensor): Input tensor to tile.

        Returns:
            torch.Tensor: Tiled version of the input tensor.
        """
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        """
        Initializes the codebook using the input x.

        Args:
            x (torch.Tensor): Input tensor to initialize the codebook.
        """
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        """
        Computes the perplexity of the codebook based on the code index.

        Args:
            code_idx (torch.Tensor): Tensor containing code indices.

        Returns:
            float: The computed perplexity.
        """
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """
        Updates the codebook based on the input x and the code index.

        Args:
            x (torch.Tensor): Input tensor to update the codebook.
            code_idx (torch.Tensor): Tensor containing code indices.
        
        Returns:
            float: The updated perplexity after codebook adjustment.
        """
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def preprocess(self, x):
        """
        Preprocesses the input tensor by permuting and reshaping it.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T).

        Returns:
            torch.Tensor: Preprocessed tensor of shape (N*T, C).
        """
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        """
        Quantizes the input tensor x by finding the closest codebook vectors.

        Args:
            x (torch.Tensor): Input tensor to quantize.

        Returns:
            torch.Tensor: Indices of the closest codebook vectors.
        """
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        """
        Dequantizes the code indices back into codebook vectors.

        Args:
            code_idx (torch.Tensor): Code indices to dequantize.

        Returns:
            torch.Tensor: Dequantized tensor from the codebook vectors.
        """
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        """
        Forward pass for quantization and dequantization through the codebook.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Dequantized tensor, commitment loss, and perplexity.
        """
        N, width, T = x.shape

        # Preprocess the input
        x = self.preprocess(x)

        # Initialize the codebook if not initialized
        if self.training and not self.init:
            self.init_codebook(x)

        # Quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update the codebook during training
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Compute the commitment loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough operation
        x_d = x + (x_d - x).detach()

        # Postprocess the output
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity