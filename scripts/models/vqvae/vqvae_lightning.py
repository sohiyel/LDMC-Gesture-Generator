import lightning as L
import torch
from models.vqvae.vqvae_2 import VQVAE_Model
from torch import optim
import utils.losses as losses
import logging
from models.motion_embedding.embedding_space_evaluator import EmbeddingSpaceEvaluator
import os
import numpy as np
from data_analysis import plot_two_pca
import pickle

class LightningVQVAE(L.LightningModule):
    """
    PyTorch Lightning module for training a VQ-VAE model for gesture generation.
    
    Attributes:
        model (VQVAE_Model): The VQ-VAE model.
        Loss (ReConsLoss): Reconstruction loss function.
        embed_space_evaluator (EmbeddingSpaceEvaluator): Evaluator for embedding space metrics.
        mean_pose, mean_dir_vec (tuple): Precomputed mean pose and direction vectors.
        commit, contrastive (float): Loss coefficients for commitment and contrastive losses.
        lr, weight_decay, gamma (float): Learning rate, weight decay, and gamma for the optimizer.
        lr_scheduler (list): Learning rate scheduler milestones.
        avg_* (float): Variables for tracking running averages of various losses and metrics.
        list_audio_embeddings, list_text_embeddings, list_gesture_*_embeddings (numpy.array): Arrays for storing embeddings.
        eval_iter (int): Interval for performing evaluation.
    """

    def __init__(self, args):
        """
        Initializes the LightningVQVAE module.

        Args:
            args: Command line arguments containing model configuration parameters.
        """
        super().__init__()
        self.save_hyperparameters()

        self.out_dir = os.path.join(args.out_dir, args.exp_name)
        self.eval_iter = args.eval_iter
        self.in_pose_dim, self.out_pose_dim, self.n_poses = args.in_pose_dim, args.out_pose_dim, args.n_poses
        self.n_pre_poses = args.n_pre_poses
        self.subject = args.subject
        
        # Initialize the VQVAE model
        self.model = VQVAE_Model(
            args.code_dim,
            args.nb_code,
            args.quantizer,
            args.in_pose_dim,
            args.out_pose_dim,
            args.down_t,
            args.stride,
            args.depth,
            args.dilation_growth_rate,
            args.temperature,
            args.batch_size,
            args,
            self.device
        )
        
        # Loss and evaluation modules
        self.Loss = losses.ReConsLoss(args.recons_loss)
        self.embed_space_evaluator = EmbeddingSpaceEvaluator(args, os.path.join(args.data_path, args.embed_evaluator_path), self.device)
        self.mean_pose, self.mean_dir_vec = pickle.load(open(args.mean_vectors_path, "rb"))
        self.commit = args.commit
        self.contrastive = args.contrastive

        # Optimizer parameters
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.gamma = args.gamma
        self.lr_scheduler = args.lr_scheduler

        # Average metrics for training and validation
        self.avg_recons = 0.
        self.avg_perplexity = 0.
        self.avg_commit = 0.
        self.avg_gesture_text = 0.
        self.avg_gesture_audio = 0.
        self.avg_gesture_style = 0.
        
        self.avg_recons_val = 0.
        self.avg_perplexity_val = 0.
        self.avg_commit_val = 0.
        self.avg_gesture_text_val = 0.
        self.avg_gesture_audio_val = 0.
        self.avg_gesture_style_val = 0.
        
        # Tracking counters
        self.train_counter = 0
        self.val_counter = 0

        # Embeddings storage
        self.list_audio_embeddings = np.empty((0, 512))
        self.list_text_embeddings = np.empty((0, 768))
        self.list_gesture_audio_embeddings = np.empty((0, 512))
        self.list_gesture_text_embeddings = np.empty((0, 512))

    def forward(self, target_vec):
        """
        Forward pass of the VQ-VAE model. Generates predicted gestures for the input vectors.

        Args:
            target_vec (torch.Tensor): Input gesture vectors.

        Returns:
            torch.Tensor: Predicted gesture vectors.
        """
        audio_embeddings = torch.zeros(target_vec.shape[0], 512).to(target_vec.device)
        text_embeddings = torch.zeros(target_vec.shape[0], 768).to(target_vec.device)
        pred_body, loss_commit_b, perplexity_b, body_text_loss, body_audio_loss = self.model(target_vec, audio_embeddings, text_embeddings)
        return pred_body

    def training_step(self, batch, batch_idx):
        """
        Training step for the VQ-VAE model. Processes a batch and computes the training loss.

        Args:
            batch (tuple): Input batch containing gesture data and embeddings.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Total loss for the batch.
        """
        _, _, _, target_vec, _, _, _, audio_embeddings, text_embeddings, video_indices = batch

        # Split body and hand tensors depending on the input size
        if target_vec.shape[-1] == 159:
            split_tensors = torch.split(target_vec, [33, 126], dim=2)
            body, hand = split_tensors
        elif target_vec.shape[-1] == 126:
            body, hand = self.split_body_hand(target_vec)

        # Get predictions and losses from the model
        pred_gesture, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, gesture_style_loss = self.model(target_vec, audio_embeddings, text_embeddings, video_indices)

        # Compute reconstruction loss based on subject (Body/Hand)
        if self.subject == 'Body':
            loss = self.Loss(pred_gesture, body)
        elif self.subject == 'Hand':
            loss = self.Loss(pred_gesture, hand)

        # Total loss combining reconstruction, commitment, and contrastive losses
        total_loss = loss + self.commit * loss_commit + self.contrastive * (gesture_text_loss + gesture_audio_loss + gesture_style_loss)

        # Update running averages for metrics
        self.update_train_averages(loss, perplexity, loss_commit, gesture_text_loss, gesture_audio_loss, gesture_style_loss)

        self.train_counter = batch_idx
        return total_loss

    def split_body_hand(self, target_vec):
        """
        Splits the target vector into body and hand components based on predefined indices.

        Args:
            target_vec (torch.Tensor): Input gesture vector.

        Returns:
            tuple: Body and hand components.
        """
        target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
        body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
        mask = np.zeros(target_vec.shape[2], dtype=bool)
        mask[body_indices] = True
        body, hand = target_vec[:, :, mask, :], target_vec[:, :, ~mask, :]
        body = body.reshape(target_vec.shape[0], target_vec.shape[1], -1)
        hand = hand.reshape(target_vec.shape[0], target_vec.shape[1], -1)
        target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1)
        return body, hand

    def update_train_averages(self, loss, perplexity, loss_commit, gesture_text_loss, gesture_audio_loss, gesture_style_loss):
        """
        Updates the running averages for various training metrics.

        Args:
            loss (torch.Tensor): Reconstruction loss.
            perplexity (torch.Tensor): Perplexity metric.
            loss_commit (torch.Tensor): Commitment loss.
            gesture_text_loss (torch.Tensor): Gesture-text similarity loss.
            gesture_audio_loss (torch.Tensor): Gesture-audio similarity loss.
            gesture_style_loss (torch.Tensor): Gesture-style similarity loss.
        """
        self.avg_recons += loss.item()
        self.avg_perplexity += perplexity.item()
        self.avg_commit += loss_commit.item()
        self.avg_gesture_text += gesture_text_loss.item()
        self.avg_gesture_audio += gesture_audio_loss.item()
        self.avg_gesture_style += gesture_style_loss.item()

    def on_train_epoch_end(self):
        """
        Callback at the end of the training epoch. Logs average metrics.
        """
        self.log_train_averages()

        # Reset average values and evaluator
        self.avg_recons, self.avg_perplexity, self.avg_commit = 0., 0., 0.
        self.avg_gesture_text, self.avg_gesture_audio, self.avg_gesture_style = 0., 0., 0.
        self.train_counter = 0
        self.embed_space_evaluator.reset()

    def log_train_averages(self):
        """
        Logs average training metrics for the current epoch.
        """
        self.avg_recons /= self.train_counter
        self.avg_perplexity /= self.train_counter
        self.avg_commit /= self.train_counter
        self.avg_gesture_text /= self.train_counter
        self.avg_gesture_audio /= self.train_counter
        self.avg_gesture_style /= self.train_counter

        log_info = f"Epoch[{self.current_epoch}]: LR {self.lr_schedulers().get_last_lr()} : Commit. {self.avg_commit:.5f} PPL. {self.avg_perplexity:.2f} "
        log_info += f"Recons.  {self.avg_recons:.5f} {self.subject}_text. {self.avg_gesture_text:.4f} {self.subject}_audio. {self.avg_gesture_audio:.4f} {self.subject}_style. {self.avg_gesture_style:.4f}"
        logging.info(log_info)

        self.log("recons_loss", self.avg_recons)
        self.log("perplexity", self.avg_perplexity)
        self.log("commit", self.avg_commit)
        self.log("gesture_text_loss", self.avg_gesture_text)
        self.log("gesture_audio_loss", self.avg_gesture_audio)
        self.log("gesture_style_loss", self.avg_gesture_style)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VQ-VAE model. Processes a batch and computes validation loss.

        Args:
            batch (tuple): Input batch containing gesture data and embeddings.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        _, _, _, target_vec, _, _, _, audio_embeddings, text_embeddings, video_indices = batch

        # Split body and hand tensors depending on the input size
        if target_vec.shape[-1] == 159:
            split_tensors = torch.split(target_vec, [33, 126], dim=2)
            body, hand = split_tensors
        elif target_vec.shape[-1] == 126:
            body, hand = self.split_body_hand(target_vec)

        # Get predictions and losses from the model
        pred_body, pred_hand = body, hand
        if self.subject == 'Body':
            pred_body, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, gesture_style_loss = self.model(target_vec, audio_embeddings, text_embeddings, video_indices)
        if self.subject == 'Hand':
            pred_hand, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, gesture_style_loss = self.model(target_vec, audio_embeddings, text_embeddings, video_indices)

        # Update validation metrics
        self.avg_gesture_text_val += gesture_text_loss.item()
        self.avg_gesture_audio_val += gesture_audio_loss.item()
        self.avg_perplexity_val += perplexity.item()
        self.avg_commit_val += loss_commit.item()
        self.avg_gesture_style_val += gesture_style_loss.item()

        # Compute reconstruction loss based on subject (Body/Hand)
        if self.subject == 'Body':
            loss_motion = self.Loss(pred_body, body)
        elif self.subject == 'Hand':
            loss_motion = self.Loss(pred_hand, hand)
        self.avg_recons_val += loss_motion.item()

        # Additional processing for combining body and hand
        if target_vec.shape[-1] == 159:
            pred_motion = torch.concat([pred_body, pred_hand], dim=-1)
        elif target_vec.shape[-1] == 126:
            pred_motion = self.reassemble_body_hand(target_vec, pred_body, pred_hand)

        self.val_counter = batch_idx
        if self.current_epoch % self.eval_iter == 1:
            gesture_embeddings = torch.nn.functional.normalize(self.model.encode(pred_motion))
            gesture_audio_embeddings, gesture_text_embeddings = gesture_embeddings[:,1,:].squeeze(-1), gesture_embeddings[:,0,:].squeeze(-1)
            self.list_gesture_audio_embeddings = np.vstack([self.list_gesture_audio_embeddings, gesture_audio_embeddings.detach().cpu().numpy()])
            self.list_gesture_text_embeddings = np.vstack([self.list_gesture_text_embeddings, gesture_text_embeddings.detach().cpu().numpy()])

            self.list_audio_embeddings = np.vstack([self.list_audio_embeddings, torch.nn.functional.normalize(audio_embeddings).squeeze(1).detach().cpu().numpy()])
            self.list_text_embeddings = np.vstack([self.list_text_embeddings, torch.nn.functional.normalize(text_embeddings).squeeze(1).detach().cpu().numpy()])

        # Push predictions to the embedding space evaluator
        if self.embed_space_evaluator:
            self.embed_space_evaluator.push_samples(_, _, pred_motion, target_vec)

    def reassemble_body_hand(self, target_vec, pred_body, pred_hand):
        """
        Reassembles the predicted body and hand components into a full motion vector.

        Args:
            target_vec (torch.Tensor): Original target gesture vector.
            pred_body (torch.Tensor): Predicted body vector.
            pred_hand (torch.Tensor): Predicted hand vector.

        Returns:
            torch.Tensor: Reassembled motion vector.
        """
        target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
        body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
        mask = np.zeros(target_vec.shape[2], dtype=bool)
        mask[body_indices] = True
        pred_motion = torch.zeros_like(target_vec)
        pred_motion[:, :, mask, :] = pred_body.reshape(pred_body.shape[0], pred_body.shape[1], -1, 3)
        pred_motion[:, :, ~mask, :] = pred_hand.reshape(pred_hand.shape[0], pred_hand.shape[1], -1, 3)
        return pred_motion.reshape(pred_motion.shape[0], pred_motion.shape[1], -1)

    def on_validation_epoch_end(self):
        """
        Callback at the end of the validation epoch. Logs validation metrics and performs evaluation.
        """
        self.log_validation_averages()

        # Reset average values for validation
        self.avg_recons_val, self.avg_perplexity_val, self.avg_commit_val = 0., 0., 0.
        self.avg_gesture_text_val, self.avg_gesture_audio_val, self.avg_gesture_style_val = 0., 0., 0.
        self.val_counter = 0

        # Perform PCA and plot embeddings
        if self.current_epoch % self.eval_iter == 1:
            self.perform_pca_plot()

    def log_validation_averages(self):
        """
        Logs average validation metrics for the current epoch.
        """
        self.avg_recons_val /= self.val_counter
        self.avg_perplexity_val /= self.val_counter
        self.avg_commit_val /= self.val_counter
        self.avg_gesture_text_val /= self.val_counter
        self.avg_gesture_audio_val /= self.val_counter
        self.avg_gesture_style_val /= self.val_counter

        frechet_dist, feat_dist = self.embed_space_evaluator.get_scores()
        diversity_score = self.embed_space_evaluator.get_diversity_scores()

        log_info = f"[VAL] FGD: {frechet_dist:.3f} feat_D: {feat_dist:.3f} diversity: {diversity_score:.2f} Commit: {self.avg_commit_val:.5f} PPL: {self.avg_perplexity_val:.2f} Recons: {self.avg_recons_val:.5f} "
        log_info += f"{self.subject}_text: {self.avg_gesture_text_val:.4f} {self.subject}_audio: {self.avg_gesture_audio_val:.4f} {self.subject}_style: {self.avg_gesture_style_val:.4f}"
        logging.info(log_info)

        self.log("FGD", frechet_dist, sync_dist=True)
        self.log("feat_D", feat_dist, sync_dist=True)
        self.log("diversity", diversity_score, sync_dist=True)
        self.log("perplexity_val", self.avg_perplexity_val, sync_dist=True)
        self.log("commit_val", self.avg_commit_val, sync_dist=True)
        self.log("gesture_text_val_loss", self.avg_gesture_text_val, sync_dist=True)
        self.log("gesture_audio_val_loss", self.avg_gesture_audio_val, sync_dist=True)
        self.log("gesture_style_val_loss", self.avg_gesture_style_val, sync_dist=True)

    def perform_pca_plot(self):
        """
        Performs PCA on embeddings and generates PCA plots for audio and text embeddings.
        """
        logging.info("Sampling...")
        out_dir = os.path.join(self.out_dir, str(self.current_epoch))
        os.makedirs(out_dir, exist_ok=True)

        plot_two_pca(self.list_audio_embeddings, self.list_gesture_audio_embeddings,
                     'Audio embeddings', f'{self.subject} joints embeddings',
                     f'Distribution of the {self.subject} joint and audio embeddings',
                     os.path.join(out_dir, f'{self.subject}_audio.png'))

        plot_two_pca(self.list_text_embeddings, self.list_gesture_text_embeddings,
                     'Text embeddings', f'{self.subject} joints embeddings',
                     f'Distribution of the {self.subject} joint and text embeddings',
                     os.path.join(out_dir, f'{self.subject}_text.png'))

        # Reset stored embeddings
        self.list_audio_embeddings = np.empty((0, 512))
        self.list_text_embeddings = np.empty((0, 768))
        self.list_gesture_audio_embeddings = np.empty((0, 512))
        self.list_gesture_text_embeddings = np.empty((0, 512))

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            tuple: List of optimizers and list of learning rate schedulers.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler, gamma=self.gamma)
        return [optimizer], [scheduler]
