import torch.nn as nn
from models.vqvae.encdec import Encoder, Decoder
from models.vqvae.quantize_cnn import QuantizeEMAReset
import torch

class VQVAE_Model(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) model.
    This model encodes input data into a latent space using a quantized codebook and decodes it back to its original form.
    
    Attributes:
        code_dim (int): Dimensionality of the code vectors in the latent space.
        num_code_b (int): Number of codebook entries.
        quant (str): Type of quantizer to use.
        encoder (Encoder): Encoder module for compressing the input.
        decoder (Decoder): Decoder module for reconstructing the input.
        quantizer (nn.Module): Quantizer for encoding and decoding vectors in the latent space.
        textLayer (nn.Linear): Linear layer to process text embeddings.
        cosine_sim (nn.CosineSimilarity): Cosine similarity module for computing similarities between embeddings.
        temperature (float): Temperature parameter for scaling the cosine similarity.
        batch_size (int): Batch size used for training the model.
    """
    
    def __init__(self,
                 code_dim,
                 nb_code,
                 quantizer,
                 in_pose_dim,
                 out_pose_dim,
                 down_t,
                 stride_t,
                 depth,
                 dilation_growth_rate,
                 temperature,
                 batch_size,
                 args,
                 device=None):
        """
        Initializes the VQVAE_Model with the specified parameters.

        Args:
            code_dim (int): Dimensionality of the latent code vectors.
            nb_code (int): Number of codebook vectors.
            quantizer (str): Type of quantizer to use ('ema_reset', 'orig', 'ema', 'reset').
            in_pose_dim (int): Input dimensionality for the pose.
            out_pose_dim (int): Output dimensionality for the pose.
            down_t (int): Number of downsampling layers.
            stride_t (int): Stride size for convolutional layers.
            depth (int): Depth of each layer.
            dilation_growth_rate (int): Growth rate for dilation in the convolutional layers.
            temperature (float): Temperature parameter for similarity computation.
            batch_size (int): Batch size for training.
            args: Additional arguments for the model configuration.
            device (torch.device, optional): Device to run the model on.
        """
        super().__init__()
        self.code_dim = code_dim
        self.num_code_b = nb_code
        self.quant = quantizer
        self.encoder = Encoder(in_pose_dim, code_dim, down_t, stride_t, nb_code, depth, dilation_growth_rate, activation='relu', norm=None)
        self.decoder = Decoder(out_pose_dim, code_dim, down_t, stride_t, nb_code, depth, dilation_growth_rate, activation='relu', norm=None)
        
        # Initialize the quantizer with EMA and codebook reset technique
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, args, device=device)

        # Initialize text layer for embedding text features
        self.textLayer = torch.nn.Linear(768, 512)
        for param in self.textLayer.parameters():
            param.requires_grad = False

        # Cosine similarity for measuring similarity between embeddings
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
        self.batch_size = batch_size

    def preprocess(self, x):
        """
        Preprocesses the input by permuting its dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, joints * 3).

        Returns:
            torch.Tensor: Permuted tensor of shape (batch_size, joints * 3, time_steps).
        """
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        """
        Postprocesses the output by permuting it back to the original format.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, joints * 3, time_steps).

        Returns:
            torch.Tensor: Permuted tensor of shape (batch_size, time_steps, joints * 3).
        """
        return x.permute(0, 2, 1)

    def encode(self, x):
        """
        Encodes the input using the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, joints * 3).

        Returns:
            torch.Tensor: Encoded tensor.
        """
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return self.postprocess(x_encoder)

    def get_code_indices(self, x):
        """
        Gets the code indices by quantizing the encoder output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, joints * 3).

        Returns:
            torch.Tensor: Code indices.
        """
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (batch_size * time_steps, code_dim)
        code_idx = self.quantizer.quantize(x_encoder)
        return code_idx.view(x.shape[0], -1)

    def create_positive_mask(self, categories):
        """
        Creates a positive mask based on categories, indicating which pairs belong to the same category.

        Args:
            categories (list): List of category labels.

        Returns:
            torch.Tensor: Positive mask of shape (num_objects, num_objects).
        """
        categories = torch.tensor(categories)
        num_objects = len(categories)
        mask = torch.zeros((num_objects, num_objects), dtype=torch.int32)
        
        for i in range(num_objects):
            mask[i] = (categories == categories[i]).int()
        
        return mask

    def get_similiarity(self, z_i, z_j, mask, temperature):
        """
        Computes the similarity between two sets of embeddings using cosine similarity.

        Args:
            z_i (torch.Tensor): First set of embeddings.
            z_j (torch.Tensor): Second set of embeddings.
            mask (torch.Tensor): Positive mask.
            temperature (float): Temperature for scaling the similarity.

        Returns:
            torch.Tensor: Computed similarity loss.
        """
        total_indices, embeddings = z_j.shape

        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        z_i = z_i[None, :, :]
        z_j = z_j[:, None, :]

        z_i = z_i.expand(total_indices, total_indices, embeddings)
        z_j = z_j.expand(total_indices, total_indices, embeddings)

        similarity_matrix = self.cosine_sim(z_i, z_j) / temperature
        positive_mask = mask
        negative_mask = 1 - positive_mask

        exp_positive_scores = torch.exp(similarity_matrix) * positive_mask
        sum_positive = exp_positive_scores.sum(dim=1, keepdim=True)

        exp_negative_scores = torch.exp(similarity_matrix) * negative_mask
        sum_negative = exp_negative_scores.sum(dim=1, keepdim=True)

        log_prob = torch.log(sum_positive / sum_negative)
        loss = - (positive_mask * log_prob).sum(dim=1)

        return loss.mean()

    def forward(self, gesture, audio_embeddings, text_embeddings, vid_indices):
        """
        Forward pass of the VQ-VAE model for computing quantization loss and gesture-audio/text similarity.

        Args:
            gesture (torch.Tensor): Input gesture tensor.
            audio_embeddings (torch.Tensor): Audio embeddings tensor.
            text_embeddings (torch.Tensor): Text embeddings tensor.
            vid_indices (list): List of video indices for style loss.

        Returns:
            tuple: Output tensor, commitment loss, perplexity, gesture-text loss, gesture-audio loss, and gesture-style loss.
        """
        audio_embeddings = audio_embeddings.reshape(-1, 512)
        text_embeddings = self.textLayer(text_embeddings).reshape(-1, 512)

        x = self.preprocess(gesture)
        x_encoder = self.encoder(x)

        gesture_embeddings1, gesture_embeddings2, gesture_embeddings3, gesture_embeddings4 = (
            x_encoder[:, :, 0].squeeze(-1),
            x_encoder[:, :, 1].squeeze(-1),
            x_encoder[:, :, 2].squeeze(-1),
            x_encoder[:, :, 3].squeeze(-1)
        )

        positive_mask = torch.eye(text_embeddings.shape[0]).to(text_embeddings.device)
        gesture_text_loss = self.get_similiarity(text_embeddings, gesture_embeddings1, positive_mask, self.temperature)
        gesture_audio_loss = self.get_similiarity(audio_embeddings, gesture_embeddings2, positive_mask, self.temperature)
        positive_mask = self.create_positive_mask(vid_indices).to(text_embeddings.device)
        gesture_style_loss_1 = self.get_similiarity(gesture_embeddings3, gesture_embeddings3, positive_mask, 1)
        gesture_style_loss_2 = self.get_similiarity(gesture_embeddings4, gesture_embeddings4, positive_mask, 1)

        x_quantized, loss_commit, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, (gesture_style_loss_1 + gesture_style_loss_2) / 2

    def forward_decoder(self, x):
        """
        Forward pass for decoding from code indices.

        Args:
            x (torch.Tensor): Input code indices.

        Returns:
            torch.Tensor: Decoded output.
        """
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        x_decoder = self.decoder(x_d)
        return self.postprocess(x_decoder)
