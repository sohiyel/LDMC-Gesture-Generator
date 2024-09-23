import lightning as L
import torch
from torch import optim
import utils.losses as losses
import logging
import numpy as np
import torch.nn as nn
from models.diffusion.diffusion import DiffusionModel
from models.vqvae.vqvae_lightning import LightningVQVAE
from models.motion_embedding.embedding_space_evaluator import EmbeddingSpaceEvaluator
import os

class LightningDiffusion(L.LightningModule):
    """
    PyTorch Lightning module for diffusion-based gesture generation, integrating a VQ-VAE model 
    for quantization and decoding.
    
    Attributes:
        subject (str): Target subject for gesture generation (e.g., 'Body' or 'Hand').
        vqvaeModel (LightningVQVAE): Pre-trained VQ-VAE model.
        diffusion (DiffusionModel): Diffusion model for gesture generation.
        lr (float): Learning rate for the optimizer.
        lr_scheduler (list): Milestones for learning rate scheduler.
        weight_decay (float): Weight decay value for the optimizer.
        gamma (float): Decay factor for the learning rate scheduler.
        loss, val_loss (list): Lists to store training and validation loss values.
        Loss (ReConsLoss): Reconstruction loss function.
        pre_pose_encoder (nn.Linear): Linear layer to encode pre-poses for conditioning.
        embed_space_evaluator (EmbeddingSpaceEvaluator): Evaluator for embedding space metrics.
    """

    def __init__(self, args):
        """
        Initializes the LightningDiffusion module, sets up VQ-VAE and diffusion models, 
        and configures training parameters.

        Args:
            args: Arguments for setting up the models and training parameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.subject = args.subject
        self.vqvaeModel = None

        # Load VQ-VAE model and initialize diffusion model
        vqvae_args = self.load_input_models(args)
        self.textLayer = self.vqvaeModel.model.textLayer
        self.diffusion = DiffusionModel(args, vqvae_args)

        # Optimizer parameters
        self.lr = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.weight_decay = args.weight_decay
        self.gamma = args.gamma

        # Loss functions and evaluation metrics
        self.loss, self.val_loss = [], []
        self.Loss = losses.ReConsLoss(args.recons_loss)
        self.pre_pose_encoder = nn.Linear(vqvae_args.n_pre_poses * vqvae_args.in_pose_dim, vqvae_args.code_dim)
        self.embed_space_evaluator = EmbeddingSpaceEvaluator(vqvae_args, os.path.join(args.data_path, args.embed_evaluator_path), self.device)

    def load_input_models(self, args):
        """
        Loads the pre-trained VQ-VAE model and sets it to evaluation mode.

        Args:
            args: Arguments containing paths to the VQ-VAE model and data paths.

        Returns:
            vqvae_args: VQ-VAE model arguments needed for the diffusion model.
        """
        ckpt = torch.load(args.vqvae_path, map_location='cpu')
        ckpt['hyper_parameters']['args'].embed_evaluator_path = args.embed_evaluator_path
        ckpt['hyper_parameters']['args'].data_path = args.data_path

        # Load the VQ-VAE model
        self.vqvaeModel = LightningVQVAE(ckpt['hyper_parameters']['args'])
        self.vqvaeModel.load_state_dict(ckpt['state_dict'])
        self.vqvaeModel.eval()

        # Disable training for the VQ-VAE model
        self.vqvaeModel.model.quantizer.training = False
        self.vqvaeModel.training = False
        for p in self.vqvaeModel.parameters():
            p.requires_grad = False

        return ckpt['hyper_parameters']['args']

    def forward(self, audio_embeddings, text_embeddings, pre_poses):
        """
        Forward pass to generate gesture predictions based on audio, text, and pre-poses.

        Args:
            audio_embeddings (torch.Tensor): Audio embeddings for conditioning.
            text_embeddings (torch.Tensor): Text embeddings for conditioning.
            pre_poses (torch.Tensor): Pre-poses for gesture guidance.

        Returns:
            torch.Tensor: Generated gesture predictions.
        """
        # Encode text and audio embeddings
        text_embeddings = self.textLayer(text_embeddings).squeeze(1)
        audio_embeddings = audio_embeddings.squeeze(1)
        pre_poses = pre_poses.reshape(audio_embeddings.shape[0], -1).to(audio_embeddings.device)
        pre_pose_embeddings = self.pre_pose_encoder(pre_poses)

        # Generate gesture embeddings using diffusion model
        conditions = [audio_embeddings, text_embeddings, pre_pose_embeddings]
        gesture_embeddings = self.diffusion.sample(self.diffusion.uNet, text_embeddings.shape[0], conditions)

        # Preprocess, quantize, and decode gesture embeddings
        gesture_embeddings = self.vqvaeModel.model.preprocess(gesture_embeddings)
        x_quantized, _, _ = self.vqvaeModel.model.quantizer(gesture_embeddings)
        x_decoder = self.vqvaeModel.model.decoder(x_quantized)
        x_out = self.vqvaeModel.model.postprocess(x_decoder)

        return x_out

    def training_step(self, batch, batch_idx):
        """
        Training step for the diffusion model. Processes a batch and computes the loss.

        Args:
            batch (tuple): Input batch containing gesture data and embeddings.
            batch_idx (int): Batch index for logging.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        # Extract inputs from the batch
        _, _, _, target_vec, _, _, _, audio_embeddings, text_embeddings, _ = batch

        # Encode text, audio, and pre-poses
        text_embeddings = self.textLayer(text_embeddings).squeeze(1)
        audio_embeddings = audio_embeddings.squeeze(1)
        pre_poses = target_vec[:, :4, :].view(target_vec.size(0), -1)
        pre_pose_embeddings = self.pre_pose_encoder(pre_poses)

        # Encode target gestures and compute loss
        gesture_embeddings = self.vqvaeModel.model.encode(target_vec)
        conditions = [audio_embeddings, text_embeddings, pre_pose_embeddings]
        loss = self.diffusion(gesture_embeddings, conditions)

        self.loss.append(loss.cpu().detach())
        return loss

    def on_train_epoch_end(self):
        """
        Callback at the end of the training epoch. Logs the average training loss.
        """
        # Compute and log average training loss
        loss = np.mean(np.array(self.loss))
        logging.info(f"Loss: {loss}, LR: {self.lr_schedulers().get_last_lr()} ")
        self.loss = []

        self.log("loss", loss)
        self.embed_space_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the diffusion model. Processes a validation batch and computes the loss.

        Args:
            batch (tuple): Input batch containing gesture data and embeddings.
            batch_idx (int): Batch index for logging.

        Returns:
            torch.Tensor: Computed validation loss for the batch.
        """
        # Extract inputs from the batch
        _, _, _, target_vec, _, _, _, audio_embeddings, text_embeddings, _ = batch

        # Split body and hand depending on input size
        if target_vec.shape[-1] == 159:
            body, hand = torch.split(target_vec, [33, 126], dim=2)
        elif target_vec.shape[-1] == 126:
            target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
            body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
            mask = np.zeros(target_vec.shape[2], dtype=bool)
            mask[body_indices] = True
            body, hand = target_vec[:, :, mask, :], target_vec[:, :, ~mask, :]
            body = body.reshape(target_vec.shape[0], target_vec.shape[1], -1)
            hand = hand.reshape(target_vec.shape[0], target_vec.shape[1], -1)
            target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1)
        pred_body, pred_hand = body, hand
        # Encode text, audio, and pre-poses
        text_embeddings = self.textLayer(text_embeddings).squeeze(1)
        audio_embeddings = audio_embeddings.squeeze(1)
        pre_poses = target_vec[:, :4, :].view(target_vec.size(0), -1).to(audio_embeddings)
        pre_pose_embeddings = self.pre_pose_encoder(pre_poses)

        # Generate gesture embeddings using diffusion model
        conditions = [audio_embeddings, text_embeddings, pre_pose_embeddings]
        predicted_embeddings = self.diffusion.sample(self.diffusion.uNet, target_vec.shape[0], conditions)

        # Preprocess and decode the predicted embeddings
        x_encoder = self.vqvaeModel.model.preprocess(predicted_embeddings)
        x_quantized, _, _ = self.vqvaeModel.model.quantizer(x_encoder)
        x_decoder = self.vqvaeModel.model.decoder(x_quantized)

        # Post-process body and hand predictions
        if self.subject == 'Body':
            pred_body = self.vqvaeModel.model.postprocess(x_decoder)
        if self.subject == 'Hand':
            pred_hand = self.vqvaeModel.model.postprocess(x_decoder)

        # Concatenate body and hand or reassemble them
        if target_vec.shape[-1] == 159:
            pred_motion = torch.concat([pred_body, pred_hand], dim=-1)
        elif target_vec.shape[-1] == 126:
            target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
            mask = np.zeros(target_vec.shape[2], dtype=bool)
            mask[body_indices] = True
            pred_motion = torch.zeros_like(target_vec)
            pred_motion[:, :, mask, :] = pred_body.reshape(pred_body.shape[0], pred_body.shape[1], -1, 3)
            pred_motion[:, :, ~mask, :] = pred_hand.reshape(pred_hand.shape[0], pred_hand.shape[1], -1, 3)
            pred_motion = pred_motion.reshape(pred_motion.shape[0], pred_motion.shape[1], -1)

        # Push predicted samples to the embedding space evaluator
        if self.embed_space_evaluator:
            self.embed_space_evaluator.push_samples(_, _, pred_motion, target_vec)

        # Compute the loss for body or hand
        if self.subject == 'Body':
            loss_motion = self.Loss(pred_body, body)
        if self.subject == 'Hand':
            loss_motion = self.Loss(pred_hand, hand)

        self.val_loss.append(loss_motion.cpu().detach())
        return loss_motion

    def on_validation_epoch_end(self):
        """
        Callback at the end of the validation epoch. Logs validation loss and metrics such as FGD and diversity.
        """
        # Compute evaluation metrics and log them
        frechet_dist, feat_dist = self.embed_space_evaluator.get_scores()
        diversity_score = self.embed_space_evaluator.get_diversity_scores()

        val_loss = np.mean(np.array(self.val_loss))
        logging.info(f"ValLoss: {val_loss} FGD: {frechet_dist:.3f}, Feature Dist: {feat_dist:.3f}, Diversity: {diversity_score:.2f}")
        self.val_loss = []

        self.log("val_loss", val_loss, sync_dist=True)
        self.log("FGD", frechet_dist, sync_dist=True)

    def test_step(self, batch, batch_idx) -> None:
        """
        Test step for the diffusion model. Processes a test batch and computes the test loss.

        Args:
            batch (tuple): Input batch containing gesture data and embeddings.
            batch_idx (int): Batch index for logging.

        Returns:
            torch.Tensor: Computed test loss for the batch.
        """
        # Similar to validation step logic
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """
        Callback at the end of the test epoch. Logs test loss and metrics such as FGD and diversity.
        """
        # Compute evaluation metrics and log them
        frechet_dist, feat_dist = self.embed_space_evaluator.get_scores()
        diversity_score = self.embed_space_evaluator.get_diversity_scores()

        val_loss = np.mean(np.array(self.val_loss))
        logging.info(f"Evaluation Loss: {val_loss} FGD: {frechet_dist:.3f}, Feature Dist: {feat_dist:.3f}, Diversity: {diversity_score:.2f}")
        self.val_loss = []

        self.log("eval_loss", val_loss, sync_dist=True)
        self.log("FGD", frechet_dist, sync_dist=True)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            list: List of optimizers and schedulers.
        """
        # AdamW optimizer and learning rate scheduler
        self.optimizer = optim.AdamW(self.diffusion.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_scheduler, gamma=self.gamma)
        return [self.optimizer], [self.scheduler]
