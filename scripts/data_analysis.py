import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.option_vq_2 import get_args_parser
from data_loader.data_loader import load_data
import torch
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

def plot_3d_scatter(vectors1, vectors2, title1='', title2='', subtitle='', save_path='scatter_3d.png'):
    """
    Plots a 3D scatter plot for two sets of vectors.

    Args:
        vectors1 (np.ndarray): First set of vectors.
        vectors2 (np.ndarray): Second set of vectors.
        title1 (str): Title for the first subplot.
        title2 (str): Title for the second subplot.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting 3D scatter plot: {subtitle}")
    x_b, y_b, z_b = vectors1[:, 0], vectors1[:, 1], vectors1[:, 2]
    x_h, y_h, z_h = vectors2[:, 0], vectors2[:, 1], vectors2[:, 2]

    fig = plt.figure(figsize=(14, 6))

    # First subplot (vectors1)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_b, y_b, z_b, c='b', marker='o')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    plt.title(title1)

    # Second subplot (vectors2)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_h, y_h, z_h, c='r', marker='o')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    plt.title(title2)

    plt.suptitle(subtitle)
    plt.savefig(save_path, dpi=300)

def plot_histogram(vectors1, vectors2, subtitle='', save_path='hist.png'):
    """
    Plots histograms for the x, y, z coordinates of two sets of vectors.

    Args:
        vectors1 (np.ndarray): First set of vectors.
        vectors2 (np.ndarray): Second set of vectors.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting histogram: {subtitle}")
    x1, y1, z1 = vectors1[:, 0], vectors1[:, 1], vectors1[:, 2]
    x2, y2, z2 = vectors2[:, 0], vectors2[:, 1], vectors2[:, 2]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Plot Dataset 1 histograms
    axs[0, 0].hist(x1, bins=30, color='b', alpha=0.7)
    axs[0, 0].set_title('X Distribution')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(y1, bins=30, color='g', alpha=0.7)
    axs[0, 1].set_title('Y Distribution')
    axs[0, 1].set_xlabel('Y')
    axs[0, 1].set_ylabel('Frequency')

    axs[0, 2].hist(z1, bins=30, color='r', alpha=0.7)
    axs[0, 2].set_title('Z Distribution')
    axs[0, 2].set_xlabel('Z')
    axs[0, 2].set_ylabel('Frequency')

    # Plot Dataset 2 histograms
    axs[1, 0].hist(x2, bins=30, color='b', alpha=0.7)
    axs[1, 0].set_title('X Distribution')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].hist(y2, bins=30, color='g', alpha=0.7)
    axs[1, 1].set_title('Y Distribution')
    axs[1, 1].set_xlabel('Y')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 2].hist(z2, bins=30, color='r', alpha=0.7)
    axs[1, 2].set_title('Z Distribution')
    axs[1, 2].set_xlabel('Z')
    axs[1, 2].set_ylabel('Frequency')

    plt.suptitle(subtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)

def plot_2d_scatter(vectors1, vectors2, title1='', title2='', subtitle='', save_path='scatter2d.png'):
    """
    Plots 2D scatter plots for two sets of vectors, with colors indicating the z-coordinate.

    Args:
        vectors1 (np.ndarray): First set of vectors.
        vectors2 (np.ndarray): Second set of vectors.
        title1 (str): Title for the first plot.
        title2 (str): Title for the second plot.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting 2D scatter plot: {subtitle}")
    if vectors1.shape[1] != 3 or vectors2.shape[1] != 3:
        raise ValueError("The datasets do not contain 3D vectors")

    x1, y1, z1 = vectors1[:, 0], vectors1[:, 1], vectors1[:, 2]
    x2, y2, z2 = vectors2[:, 0], vectors2[:, 1], vectors2[:, 2]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # First scatter plot
    scatter1 = axs[0].scatter(x1, y1, c=z1, cmap='viridis', marker='o')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(title1)
    cbar1 = plt.colorbar(scatter1, ax=axs[0], label='Z')

    # Second scatter plot
    scatter2 = axs[1].scatter(x2, y2, c=z2, cmap='plasma', marker='o')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(title2)
    cbar2 = plt.colorbar(scatter2, ax=axs[1], label='Z')

    plt.suptitle(subtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)

def plot_tsne(embeddings, title='', save_path='tsne.png'):
    """
    Plots t-SNE for a set of embeddings.

    Args:
        embeddings (np.ndarray): Embeddings to plot.
        title (str): Title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting TSNE: {title}")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig(save_path, dpi=300)

def plot_two_tsne(embeddings1, embeddings2, label1='', label2='', subtitle='', save_path='tsne.png'):
    """
    Plots t-SNE for two sets of embeddings.

    Args:
        embeddings1 (np.ndarray): First set of embeddings.
        embeddings2 (np.ndarray): Second set of embeddings.
        label1 (str): Label for the first set.
        label2 (str): Label for the second set.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting 2 TSNE: {subtitle}")
    common_dim = 50
    pca_512 = PCA(n_components=common_dim)
    pca_768 = PCA(n_components=common_dim)

    embeddings1_reduced = pca_512.fit_transform(embeddings1)
    embeddings2_reduced = pca_768.fit_transform(embeddings2)

    combined_embeddings = np.vstack((embeddings1_reduced, embeddings2_reduced))
    tsne = TSNE(n_components=2, random_state=42)
    combined_embeddings_2d = tsne.fit_transform(combined_embeddings)

    embeddings1_2d = combined_embeddings_2d[:len(embeddings1)]
    embeddings2_2d = combined_embeddings_2d[len(embeddings1):]

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], s=5, alpha=0.7, label=label1)
    plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], s=5, alpha=0.7, label=label2, color='r')
    plt.title(subtitle)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300)

def plot_pca(embeddings, title='', save_path='pca.png'):
    """
    Plots PCA for a set of embeddings.

    Args:
        embeddings (np.ndarray): Embeddings to plot.
        title (str): Title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting PCA: {title}")
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.savefig(save_path, dpi=300)

def plot_two_pca(embeddings1, embeddings2, title1='', title2='', subtitle='', save_path='pca.png'):
    """
    Plots PCA for two sets of embeddings.

    Args:
        embeddings1 (np.ndarray): First set of embeddings.
        embeddings2 (np.ndarray): Second set of embeddings.
        title1 (str): Title for the first set of embeddings.
        title2 (str): Title for the second set of embeddings.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting 2 PCAs: {subtitle}")
    common_dim = 2
    pca_512 = PCA(n_components=common_dim)
    pca_768 = PCA(n_components=common_dim)

    embeddings1_reduced = pca_512.fit_transform(embeddings1)
    embeddings2_reduced = pca_768.fit_transform(embeddings2)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings2_reduced[:, 0], embeddings2_reduced[:, 1], s=5, alpha=0.5, label=title2, color='r')
    plt.scatter(embeddings1_reduced[:, 0], embeddings1_reduced[:, 1], s=5, alpha=0.3, label=title1)
    plt.title(subtitle)
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300)

def plot_three_pca(embeddings1, embeddings2, embeddings3, title1='', title2='', title3='', subtitle='', save_path='pca.png'):
    """
    Plots PCA for three sets of embeddings.

    Args:
        embeddings1 (np.ndarray): First set of embeddings.
        embeddings2 (np.ndarray): Second set of embeddings.
        embeddings3 (np.ndarray): Third set of embeddings.
        title1 (str): Title for the first set.
        title2 (str): Title for the second set.
        title3 (str): Title for the third set.
        subtitle (str): Overall title for the plot.
        save_path (str): Path to save the plot.
    """
    logging.info(f"Plotting 3 PCAs: {subtitle}")
    common_dim = 2
    pca_1 = PCA(n_components=common_dim)
    pca_2 = PCA(n_components=common_dim)
    pca_3 = PCA(n_components=common_dim)

    embeddings1_reduced = pca_1.fit_transform(embeddings1)
    embeddings2_reduced = pca_2.fit_transform(embeddings2)
    embeddings3_reduced = pca_3.fit_transform(embeddings3)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings1_reduced[:, 0], embeddings1_reduced[:, 1], s=5, alpha=0.5, label=title1)
    plt.scatter(embeddings2_reduced[:, 0], embeddings2_reduced[:, 1], s=5, alpha=0.5, label=title2, color='r')
    plt.scatter(embeddings3_reduced[:, 0], embeddings3_reduced[:, 1], s=5, alpha=0.5, label=title3, color='g')
    plt.title(subtitle)
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300)

if __name__ == '__main__':
    # Load the arguments from the parser and create output directories
    args = get_args_parser()
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the data using the provided loader
    train_loader, val_loader, test_loader = load_data(args)

    # Lists for storing various embeddings and vectors
    bodys, hands, gestures, audio_embeddings, text_embeddings = [], [], [], [], []

    logging.info("Processing the dataset.")
    for iter_idx, data in enumerate(test_loader, 0):
        _, _, _, target_vec, _, _, _, audio_embedding, text_embedding = data
        target_vec = target_vec.reshape(-1, args.in_pose_dim)
        gestures.append(target_vec.numpy())
        
        if args.in_pose_dim == 159:
            body, hand = torch.split(target_vec, [33, 126], dim=1)
            body, hand = body.reshape(-1, 3), hand.reshape(-1, 3)
        elif args.in_pose_dim == 126:
            target_vec = target_vec.reshape(target_vec.shape[0], 42, 3)
            body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
            mask = np.zeros(target_vec.shape[1], dtype=bool)
            mask[body_indices] = True
            body, hand = target_vec[:, mask, :], target_vec[:, ~mask, :]
            body, hand = body.reshape(-1, 3), hand.reshape(-1, 3)
            target_vec = target_vec.reshape(target_vec.shape[0], 126)
        bodys.append(body.numpy())
        hands.append(hand.numpy())
        audio_embeddings.append(audio_embedding.numpy())
        text_embeddings.append(text_embedding.numpy())

    gestures = np.array(gestures)
    if args.in_pose_dim == 159:
        body_sequences, hand_sequences = np.split(gestures, [33], axis=2)
        body_sequences = body_sequences.reshape(-1, 34 * 33)
        hand_sequences = hand_sequences.reshape(-1, 34 * 126)
    elif args.in_pose_dim == 126:
        temp = gestures.reshape(gestures.shape[0], gestures.shape[1], 42, 3)
        body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
        mask = np.zeros(42, dtype=bool)
        mask[body_indices] = True
        body, hand = temp[:, :, mask, :], temp[:, :, ~mask, :]
        body_sequences = body.reshape(-1, 34 * 36)
        hand_sequences = hand.reshape(-1, 34 * 90)

    gesture_sequences = gestures.reshape(-1, 34 * args.in_pose_dim)
    gestures = gestures.reshape(-1, args.in_pose_dim)
    bodys, hands = np.array(bodys).reshape(-1, 3), np.array(hands).reshape(-1, 3)
    text_embeddings = np.array(text_embeddings).reshape(-1, text_embeddings.shape[-1])
    audio_embeddings = np.array(audio_embeddings).reshape(-1, audio_embeddings.shape[-1])

    # Plot the various visualizations
    plot_2d_scatter(bodys, hands, '2D Scatter Plot - Body joints', '2D Scatter Plot - Hand joints',
                    '2D Scatter of body and hand joint direction vectors', os.path.join(args.out_dir, 'scatter_body_hand.png'))
    plot_3d_scatter(bodys, hands, '3D Scatter plot - Body joints', '3D Scatter plot - Hand joints',
                    '3D Scatter of body and hand joint direction vectors', os.path.join(args.out_dir, 'scatter_3d_body_hand.png'))
    plot_histogram(bodys, hands, 'Histograms of body and hand joint direction vectors', os.path.join(args.out_dir, 'hist_body_hand.png'))
    plot_pca(gestures, 'PCA - Gesture distributions', os.path.join(args.out_dir, 'pca_gesture.png'))
    plot_pca(gesture_sequences, 'PCA - Gesture sequences', os.path.join(args.out_dir, 'pca_gesture_seq.png'))
    plot_pca(body_sequences, 'PCA - Body joints sequences', os.path.join(args.out_dir, 'pca_body_seq.png'))
    plot_pca(hand_sequences, 'PCA - Hand joints sequences', os.path.join(args.out_dir, 'pca_hand_seq.png'))
    plot_pca(audio_embeddings, 'PCA - Audio embeddings', os.path.join(args.out_dir, 'pca_audio.png'))
    plot_pca(text_embeddings, 'PCA - Text embeddings', os.path.join(args.out_dir, 'pca_text.png'))
    plot_two_pca(audio_embeddings, text_embeddings, 'Audio embeddings', 'Text embeddings',
                 'PCA of audio and text embeddings', os.path.join(args.out_dir, 'pca_audio_text.png'))
    plot_two_pca(body_sequences, text_embeddings, 'Body joints sequences', 'Text embeddings',
                 'PCA of body joints sequences and text embeddings', os.path.join(args.out_dir, 'pca_body_text.png'))
    plot_two_pca(hand_sequences, text_embeddings, 'Hand joints sequences', 'Text embeddings',
                 'PCA of hand joints sequences and text embeddings', os.path.join(args.out_dir, 'pca_hand_text.png'))
    plot_two_pca(body_sequences, audio_embeddings, 'Body joints sequences', 'Audio embeddings',
                 'PCA of body joints sequences and audio embeddings', os.path.join(args.out_dir, 'pca_body_audio.png'))
    plot_two_pca(hand_sequences, audio_embeddings, 'Hand joints sequences', 'Audio embeddings',
                 'PCA of hand joints sequences and audio embeddings', os.path.join(args.out_dir, 'pca_hand_audio.png'))
    plot_two_pca(hand_sequences, body_sequences, 'Hand joints sequences', 'Body joints sequences',
                 'PCA of hand and body joints sequences', os.path.join(args.out_dir, 'pca_hand_body.png'))
    plot_tsne(gestures, 't-SNE - Gesture distributions', os.path.join(args.out_dir, 'tsne_gesture.png'))
    plot_tsne(gesture_sequences, 't-SNE - Gesture sequences', os.path.join(args.out_dir, 'tsne_gesture_seq.png'))
    plot_tsne(audio_embeddings, 't-SNE - Audio embeddings', os.path.join(args.out_dir, 'tsne_audio.png'))
    plot_tsne(text_embeddings, 't-SNE - Text embeddings', os.path.join(args.out_dir, 'tsne_text.png'))
    plot_two_tsne(audio_embeddings, text_embeddings, 'Audio embeddings', 'Text embeddings',
                  't-SNE of audio and text embeddings', os.path.join(args.out_dir, 'tsne_audio_text.png'))
