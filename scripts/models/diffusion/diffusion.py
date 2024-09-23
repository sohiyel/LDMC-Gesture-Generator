import torch
import utils.losses as losses
from models.diffusion.modules import UNet_conditional, EMA
import logging
import copy
import numpy as np
import torch.nn as nn
from utils.beta_schedulers import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

class DiffusionModel(nn.Module):
    """
    DiffusionModel class implements a diffusion-based generative model using a UNet architecture
    with conditional embeddings and Exponential Moving Average (EMA) for model updates.

    Attributes:
        uNet (UNet_conditional): The UNet model for generating denoised samples.
        ema (EMA): Exponential Moving Average for the model weights.
        ema_model (UNet_conditional): Copy of the UNet model with EMA-applied weights.
        nb_steps (int): The number of diffusion steps.
        betas (torch.Tensor): Diffusion beta schedule values.
        alpha (torch.Tensor): Alpha values derived from betas.
        alpha_hat (torch.Tensor): Cumulative product of alpha values for each timestep.
        nb_poses (int): Number of poses in the input gesture data.
        code_dim (int): Dimensionality of the gesture embedding codes.
        Loss (losses.ReConsLoss): The reconstruction loss function.
    """
    
    def __init__(self, args, vqvae_args):
        """
        Initializes the DiffusionModel with the given arguments, including beta schedules
        and the UNet architecture for diffusion steps.

        Args:
            args: Arguments including nb_steps and beta_schedule type.
            vqvae_args: VQ-VAE related arguments such as down_t and n_poses.
        """
        super().__init__()
        
        # Number of indices based on VQ-VAE downsampling
        self.nb_indices = 2 ** (5 - vqvae_args.down_t)

        # Conditional UNet for generating gesture embeddings
        self.uNet = UNet_conditional(c_in=self.nb_indices, c_out=self.nb_indices, time_dim=1536, num_classes=1024).requires_grad_(True)

        # EMA for stabilizing training
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.uNet).eval().requires_grad_(False)  # Copy UNet for EMA

        # Number of diffusion steps
        self.nb_steps = args.nb_steps

        # Set up the beta schedule
        if args.beta_schedule == 'linear':
            self.register_buffer("betas", linear_beta_schedule(self.nb_steps))
        elif args.beta_schedule == 'cosine':
            self.register_buffer("betas", cosine_beta_schedule(self.nb_steps))
        elif args.beta_schedule == 'sigmoid':
            self.register_buffer("betas", sigmoid_beta_schedule(self.nb_steps))
        else:
            raise ValueError(f'unknown beta schedule {args.beta_schedule}')
        
        # Compute alpha and alpha_hat from betas
        self.register_buffer("alpha", torch.Tensor(1. - self.betas))
        self.register_buffer("alpha_hat", torch.Tensor(torch.cumprod(self.alpha, dim=0)))

        # Additional parameters for pose embeddings
        self.nb_poses = vqvae_args.n_poses
        self.code_dim = vqvae_args.code_dim

        # Loss function for reconstruction
        self.Loss = losses.ReConsLoss(args.recons_loss)

    def noise_gestures(self, x, t):
        """
        Adds noise to the gesture embeddings based on the diffusion step.

        Args:
            x (torch.Tensor): Input gesture embeddings.
            t (torch.Tensor): Time step tensor.

        Returns:
            Tuple of noisy embeddings and the added noise.
        """
        sqrt_alpha_hat = torch.Tensor(torch.sqrt(self.alpha_hat[t])[:, None, None]).to(x)
        sqrt_one_minus_alpha_hat = torch.Tensor(torch.sqrt(1 - self.alpha_hat[t])[:, None, None]).to(x)
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timesteps(self, n):
        """
        Samples random timesteps for diffusion.

        Args:
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Randomly sampled timesteps.
        """
        return torch.randint(low=1, high=self.nb_steps, size=(n,))
    
    def sample(self, model, batch_size, conditions, cfg_scale=0.5, device=None):
        """
        Samples from the model using reverse diffusion to generate new gesture embeddings.

        Args:
            model: The diffusion model for generating samples.
            batch_size (int): Number of samples to generate.
            conditions (torch.Tensor): Conditional inputs (e.g., pose conditions).
            cfg_scale (float): Classifier-free guidance scale.
            device: Device to run the model on.

        Returns:
            torch.Tensor: Generated gesture embeddings.
        """
        logging.info(f"Sampling {batch_size} samples...")
        model.eval()  # Set model to evaluation mode

        # Concatenate the conditions
        conditions = torch.concatenate([conditions[0], conditions[1], conditions[2]], dim=-1)

        # Generate samples with reverse diffusion
        with torch.no_grad():
            x_t = torch.randn((batch_size, self.nb_indices, self.code_dim)).to(conditions[0].device)
            for i in reversed(range(1, self.nb_steps)):
                t = torch.Tensor((torch.ones(batch_size) * i)).to(conditions[0].device).long()
                predicted_noise = model(x_t, t, conditions)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x_t, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # Apply reverse diffusion step
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.betas[t][:, None, None]
                noise = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)
                x_t = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()  # Return model to training mode

        # Normalize the generated embeddings
        x_t = torch.nn.functional.normalize(x_t)
        return x_t
    
    def forward(self, gestureEmbeddings, conditions):
        """
        Forward pass through the diffusion model with added noise and conditional inputs.

        Args:
            gestureEmbeddings (torch.Tensor): Input gesture embeddings.
            conditions (torch.Tensor): Conditional inputs for the model.

        Returns:
            torch.Tensor: Reconstruction loss for training.
        """
        # Sample random timesteps for diffusion
        t = self.sample_timesteps(gestureEmbeddings.shape[0])
        t = torch.Tensor(t).to(gestureEmbeddings).long()

        # Add noise to the gesture embeddings
        x_t, noise = self.noise_gestures(gestureEmbeddings, t)

        # Apply random masking to the conditions for classifier-free guidance
        rand = np.random.random()
        zeroTensor = torch.zeros_like(conditions[0])
        if rand < 0.1:
            conditions = None
        elif rand < 0.2:
            conditions = torch.concatenate([conditions[0], conditions[1], zeroTensor], dim=-1)
        elif rand < 0.3:
            conditions = torch.concatenate([conditions[0], zeroTensor, conditions[2]], dim=-1)
        elif rand < 0.4:
            conditions = torch.concatenate([zeroTensor, conditions[1], conditions[2]], dim=-1)
        else:
            conditions = torch.concatenate([conditions[0], conditions[1], conditions[2]], dim=-1)

        # Predict the noise and compute the loss
        predicted_noise = self.uNet(x_t, t, conditions)
        loss = self.Loss(noise, predicted_noise)

        # Update the EMA model
        self.ema.step_ema(self.ema_model, self.uNet)

        return loss
