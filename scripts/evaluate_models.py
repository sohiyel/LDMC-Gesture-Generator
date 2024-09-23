from models.vqvae.vqvae_lightning import LightningVQVAE
from models.diffusion.diffusion_lightning import LightningDiffusion
from data_loader.data_preprocessor import DataPreprocessor
from models.motion_embedding.embedding_space_evaluator import EmbeddingSpaceEvaluator
from data_loader.data_loader import load_data
from utils.option_video import get_args_parser
from utils.train_utils import set_logger
import logging
import torch
import pickle
import os
import numpy as np
import tqdm

def main(models, data_loader, embed_space_evaluator, args, device):
    """
    Main evaluation function to process the data using the VQ-VAE or Diffusion models.
    It computes various metrics, generates predictions, and logs the evaluation results.

    Args:
        models (dict): Dictionary containing the body and/or hand models (VQ-VAE or Diffusion).
        data_loader (DataLoader): DataLoader containing the test dataset.
        embed_space_evaluator (EmbeddingSpaceEvaluator): Evaluator for embedding space metrics.
        args: Parsed arguments containing evaluation configurations.
        device (torch.device): Device for computation (CPU/GPU).
    """
    total_code_indices_body = torch.Tensor().to(device)
    total_code_indices_hand = torch.Tensor().to(device)

    loss_commit_list, perplexity_list, gesture_audio_list, gesture_text_list, gesture_style_list = [], [], [], [], []

    # Iterate over the batches of data
    for iter_idx, batch in tqdm.tqdm(enumerate(data_loader, 0)):
        _, _, _, target_vec, _, _, _, audio_embeddings, text_embeddings, video_indices = batch
        text_embeddings = text_embeddings.to(device)
        audio_embeddings = audio_embeddings.to(device)
        video_indices = video_indices.to(device)
        target_vec = target_vec.to(device)

        # Handle the shape of the target vectors (159 for body + hand, 126 for body + hand split)
        if target_vec.shape[-1] == 159:
            split_tensors = torch.split(target_vec, [33, 126], dim=2)
            body, hand = split_tensors
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

        # Model type: Diffusion or VQ-VAE
        if args.subject in ['body', 'both']:
            if args.model_type == 'diffusion':
                # Diffusion-based generation
                text_embeddings_b = models['body'].textLayer(text_embeddings).squeeze(1)
                audio_embeddings = audio_embeddings.squeeze(1)
                pre_poses = target_vec[:, :4, :].view(target_vec.size(0), -1).to(audio_embeddings)
                pre_pose_embeddings = models['body'].pre_pose_encoder(pre_poses)
                conditions = [audio_embeddings, text_embeddings_b, pre_pose_embeddings]
                predicted_embeddings = models['body'].diffusion.sample(models['body'].diffusion.uNet, target_vec.shape[0], conditions)

                # Encoding and quantizing the predicted embeddings
                x_encoder = models['body'].vqvaeModel.model.preprocess(predicted_embeddings)
                x_quantized, _, _  = models['body'].vqvaeModel.model.quantizer(x_encoder)

                x_encoder = models['body'].vqvaeModel.model.postprocess(x_encoder)
                x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
                code_idx = models['body'].vqvaeModel.model.quantizer.quantize(x_encoder)
                code_idx = code_idx.view(target_vec.shape[0], -1)
                total_code_indices_body = torch.concat((total_code_indices_body, code_idx), dim=0)

                # Decode the quantized embeddings
                x_decoder = models['body'].vqvaeModel.model.decoder(x_quantized)
                pred_body = models['body'].vqvaeModel.model.postprocess(x_decoder)

            elif args.model_type == 'vqvae':
                # VQ-VAE-based generation
                pred_body, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, gesture_style_loss = models['body'](target_vec, audio_embeddings, text_embeddings, video_indices)
                loss_commit_list.append(loss_commit.item())
                perplexity_list.append(perplexity.item())
                gesture_audio_list.append(gesture_audio_loss.item())
                gesture_text_list.append(gesture_text_loss.item())
                gesture_style_list.append(gesture_style_loss.item())

        # Repeat the process for hand if required
        if args.subject in ['hand', 'both']:
            if args.model_type == 'diffusion':
                text_embeddings_h = models['hand'].textLayer(text_embeddings).squeeze(1)
                audio_embeddings = audio_embeddings.squeeze(1)
                pre_poses = target_vec[:, :4, :].view(pre_poses.size(0), -1).to(audio_embeddings)
                pre_pose_embeddings = models['hand'].pre_pose_encoder(pre_poses)
                conditions = [audio_embeddings, text_embeddings_h, pre_pose_embeddings]
                predicted_embeddings = models['hand'].diffusion.sample(models['hand'].diffusion.uNet, target_vec.shape[0], conditions)

                x_encoder = models['hand'].vqvaeModel.model.preprocess(predicted_embeddings)
                x_quantized, _, _  = models['hand'].vqvaeModel.model.quantizer(x_encoder)

                x_encoder = models['hand'].vqvaeModel.model.postprocess(x_encoder)
                x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
                code_idx = models['hand'].vqvaeModel.model.quantizer.quantize(x_encoder)
                code_idx = code_idx.view(target_vec.shape[0], -1)
                total_code_indices_hand = torch.concat((total_code_indices_hand, code_idx), dim=0)

                x_decoder = models['hand'].vqvaeModel.model.decoder(x_quantized)
                pred_hand = models['hand'].vqvaeModel.model.postprocess(x_decoder)

            elif args.model_type == 'vqvae':
                pred_hand, loss_commit, perplexity, gesture_text_loss, gesture_audio_loss, gesture_style_loss = models['body'](target_vec, audio_embeddings, text_embeddings, video_indices)
                loss_commit_list.append(loss_commit)
                perplexity_list.append(perplexity)
                gesture_audio_list.append(gesture_audio_loss)
                gesture_text_list.append(gesture_text_loss)
                gesture_style_list.append(gesture_style_loss)

        # Combine body and hand predictions
        if target_vec.shape[-1] == 159:
            pred_motion = torch.concat([pred_body, pred_hand], dim=-1)
        elif target_vec.shape[-1] == 126:
            target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
            body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
            mask = np.zeros(target_vec.shape[2], dtype=bool)
            mask[body_indices] = True
            pred_motion = torch.zeros_like(target_vec)
            pred_motion[:, :, mask, :] = pred_body.reshape(pred_body.shape[0], pred_body.shape[1], -1, 3)
            pred_motion[:, :, ~mask, :] = pred_hand.reshape(pred_hand.shape[0], pred_hand.shape[1], -1, 3)
            target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1)
            pred_motion = pred_motion.reshape(pred_motion.shape[0], pred_motion.shape[1], -1)

        embed_space_evaluator.push_samples(_, _, pred_motion, target_vec)

    # Compute evaluation metrics
    frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    diversity_score = embed_space_evaluator.get_diversity_scores()
    commit_loss = np.mean(np.array(loss_commit_list))
    perplexity_loss = np.mean(np.array(perplexity_list))
    audio_loss = np.mean(np.array(gesture_audio_list))
    text_loss = np.mean(np.array(gesture_text_list))
    style_loss = np.mean(np.array(gesture_style_list))

    # Log the evaluation results
    logging.info(f"FGD: {frechet_dist:.3f}, Feature Dist: {feat_dist:.3f}, Diversity: {diversity_score:.2f}")
    if args.model_type == 'vqvae':
        logging.info(f"Commit: {commit_loss:.5f}, Perplexity: {perplexity_loss:.2f}, Audio: {audio_loss:.2f}, Text: {text_loss:.2f}, Style: {style_loss:.2f}")

    # Save code indices
    with open(os.path.join(args.out_dir, 'code_indices_body.p'), 'wb') as f:
        pickle.dump(np.array(total_code_indices_body.cpu()), f)
    with open(os.path.join(args.out_dir, 'code_indices_hand.p'), 'wb') as f:
        pickle.dump(np.array(total_code_indices_hand.cpu()), f)

if __name__ == "__main__":
    args = get_args_parser()
    set_logger(args.out_dir, os.path.basename(__file__).replace('.py', '.log'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = args.dataset.split("/")[0]
    model_type = args.model_type
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    models = {}
    logging.info("Loading models...")

    # Load VQ-VAE or Diffusion models
    if model_type == 'vqvae':
        if args.subject in ['body', 'both']:
            logging.info(f"Loading VQ-VAE body model from {args.body_vqvae_path}")
            ckpt_body = torch.load(args.body_vqvae_path, map_location='cpu')
            ckpt_body["hyper_parameters"]["args"].data_path = args.data_path
            model_body = LightningVQVAE(ckpt_body["hyper_parameters"]["args"])
            model_body.load_state_dict(ckpt_body['state_dict'])
            model_body.eval()
            model_body.to(device)
            models['body'] = model_body

        if args.subject in ['hand', 'both']:
            logging.info(f"Loading VQ-VAE hand model from {args.hand_vqvae_path}")
            ckpt_hand = torch.load(args.hand_vqvae_path, map_location='cpu')
            ckpt_hand["hyper_parameters"]["args"].data_path = args.data_path
            model_hand = LightningVQVAE(ckpt_hand["hyper_parameters"]["args"])
            model_hand.load_state_dict(ckpt_hand['state_dict'])
            model_hand.eval()
            model_hand.to(device)
            models['hand'] = model_hand

    elif model_type == 'diffusion':
        if args.subject in ['body', 'both']:
            logging.info(f"Loading Diffusion body model from {args.body_diffusion_path}")
            ckpt_body = torch.load(args.body_diffusion_path, map_location='cpu')
            ckpt_body["hyper_parameters"]["args"].data_path = args.data_path
            ckpt_body["hyper_parameters"]["args"].vqvae_path = args.body_vqvae_path
            model_body = LightningDiffusion(ckpt_body["hyper_parameters"]["args"])
            model_body.load_state_dict(ckpt_body['state_dict'])
            model_body.eval()
            model_body.to(device)
            models['body'] = model_body

        if args.subject in ['hand', 'both']:
            logging.info(f"Loading Diffusion hand model from {args.hand_diffusion_path}")
            ckpt_hand = torch.load(args.hand_diffusion_path, map_location='cpu')
            ckpt_hand["hyper_parameters"]["args"].data_path = args.data_path
            ckpt_hand["hyper_parameters"]["args"].vqvae_path = args.hand_vqvae_path
            model_hand = LightningDiffusion(ckpt_hand["hyper_parameters"]["args"])
            model_hand.load_state_dict(ckpt_hand['state_dict'])
            model_hand.eval()
            model_hand.to(device)
            models['hand'] = model_hand

    # Load the mean direction vector
    logging.info(f"Loading mean direction vector from {args.mean_vectors_path}")
    mean_pose, mean_dir_vec = pickle.load(open(args.mean_vectors_path, "rb"))
    data_loader, val_loader, test_loader = load_data(args)

    # Load dataset-specific configurations
    if dataset == 'aqgt':
        mean_dir_vec = mean_dir_vec.reshape(-1, 53, 3)

    elif dataset == 'TedExpressive':
        mean_dir_vec = np.array(mean_dir_vec).reshape(42, 3)
        mean_pose = np.array(mean_pose).reshape(43, 3)

    # Initialize embedding space evaluator
    logging.info("Loading embedding space evaluator")
    embed_space_evaluator = EmbeddingSpaceEvaluator(args, os.path.join(args.data_path, args.embed_evaluator_path), device)

    # Start the evaluation
    logging.info("Starting the evaluation...")
    main(models, val_loader, embed_space_evaluator, args, device)
