import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
import time

from collections import Counter
from models.vqvae.vqvae_lightning import LightningVQVAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from textwrap import wrap
from utils.data_utils import convert_dir_vec_to_pose, dir_vec_pairs_expressive

nb_code = 512

def top_indices(code_indices, num, out_dir):
    """
    Finds the most common codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        num (int): Number of top indices to return.
        out_dir (str): Directory to save results.

    Returns:
        list: List of tuples with the most common indices and their counts.
    """
    counter = Counter(code_indices.flatten())
    print("Most common indices: ", counter.most_common(num))
    return counter.most_common(num)

def plot_histogram(code_indices, dataset, out_dir):
    """
    Plots a histogram of the codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        dataset (str): Dataset name.
        out_dir (str): Directory to save the plot.
    """
    plt.clf()
    _ = plt.hist(code_indices.flatten(), bins=nb_code)
    plt.title(f"Histogram of codebook indices ({dataset.upper()})")
    plt.savefig(os.path.join(out_dir, "code_indices.png"))

def top_mixtures(code_indices, num, out_dir):
    """
    Finds the most common mixtures of codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        num (int): Number of top mixtures to return.
        out_dir (str): Directory to save results.

    Returns:
        list: List of tuples with the most common mixtures and their counts.
    """
    string_array = code_indices.astype(str)
    added_array = np.array([','.join(row) for row in string_array])
    plt.clf()
    _ = plt.hist(added_array, bins=nb_code)
    plt.title(f"Histogram of codebook mixtures ({dataset.upper()})")
    plt.savefig(os.path.join(out_dir, "code_mixtures.png"))

    counter = Counter(added_array.flatten())
    print("Most common mixture of indices: ", counter.most_common(num))
    return counter.most_common(num)

def plot_histogram_of_audio(code_indices, dataset, out_dir):
    """
    Plots a histogram of audio-aligned codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        dataset (str): Dataset name.
        out_dir (str): Directory to save the plot.
    """
    code_indices = code_indices[:, 1]
    plt.clf()
    _ = plt.hist(code_indices.flatten(), bins=nb_code)
    plt.title(f"Histogram of audio aligned indices ({dataset.upper()})")
    plt.savefig(os.path.join(out_dir, "audio_code_indices.png"))

def plot_histogram_of_text(code_indices, dataset, out_dir):
    """
    Plots a histogram of text-aligned codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        dataset (str): Dataset name.
        out_dir (str): Directory to save the plot.
    """
    code_indices = code_indices[:, 0]
    plt.clf()
    _ = plt.hist(code_indices.flatten(), bins=nb_code)
    plt.title(f"Histogram of transcript aligned indices ({dataset.upper()})")
    plt.savefig(os.path.join(out_dir, "text_code_indices.png"))

def plot_histogram_of_style(code_indices, dataset, out_dir):
    """
    Plots a histogram of style-aligned codebook indices.

    Args:
        code_indices (numpy.ndarray): Array of codebook indices.
        dataset (str): Dataset name.
        out_dir (str): Directory to save the plot.
    """
    plt.clf()
    code_indices = code_indices[:, 2:]
    _ = plt.hist(code_indices.flatten(), bins=nb_code)
    plt.title(f"Histogram of style aligned indices ({dataset.upper()})")
    plt.savefig(os.path.join(out_dir, "style_code_indices.png"))

def create_video_and_save(save_path, iter_idx, prefix, output, mean_data, title):
    """
    Creates a video of the generated gesture poses.

    Args:
        save_path (str): Path to save the video.
        iter_idx (int): Iteration index for naming the file.
        prefix (str): Prefix for the file name.
        output (numpy.ndarray): Generated output poses.
        mean_data (numpy.ndarray): Mean pose data for un-normalization.
        title (str): Title for the video.
    """
    print('Rendering a video...')
    os.makedirs(save_path, exist_ok=True)
    start = time.time()

    fig = plt.figure(figsize=(4, 4))
    axes = [fig.add_subplot(1, 1, 1, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    fig.suptitle('\n'.join(wrap(title, 75)), fontsize='medium')

    # Un-normalization and convert to poses
    mean_data = mean_data.flatten()
    if output.shape[-1] == 33:
        zeros = np.zeros((1, 34, 53, 3))
        output = output.reshape(1, 34, 11, 3)
        zeros = zeros.reshape(1, 34, 159)
    elif output.shape[-1] == 36:
        zeros = np.zeros((1, 34, 42, 3))
        body_indices = [0, 1, 2, 3, 4, 20, 21, 37, 38, 39, 40, 41]
        body_joints = output.reshape(1, -1, 12, 3)
        mean_joints = mean_data.reshape(1, 42, 3)
        for i in range(34):
            for j in range(len(body_indices)):
                zeros[0, i, body_indices[j]] = body_joints[0,i,j]
        zeros = zeros.reshape(1, 34, 126)

    output = zeros + mean_data
    if output.shape[-1] == 159:
        output_poses = convert_dir_vec_to_pose(output.reshape(-1, 159))
    elif output.shape[-1] == 126:
        output_poses = convert_dir_vec_to_pose(output.reshape(-1, 126))

    # Animation function for generating frames
    def animate(i):
        name = 'generated'
        pose = output_poses[i]
        if pose is not None:
            if pose.shape[0] == 159 or pose.shape[0] == 126:
                pose = np.reshape(pose, (-1, 3))
            axes[0].clear()
            if pose.shape[0] == 53:
                ubs = 6

                skeleton_parents = np.asarray(
                    [-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
                hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, -4, 13, 14, 15, -4, 17, 18, 19])
                hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, -22, 13, 14, 15, -22, 17, 18, 19])
                hand_parents_l = hand_parents_l + 17 - ubs
                hand_parents_r = hand_parents_r + 17 + 21 - ubs

                skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)
                for j, j_parent in enumerate(skeleton_parents):
                    if j_parent == -1:
                        continue
                    multi = 2
                    axes[0].plot([pose[j, 0] * multi, pose[j_parent, 0] * multi],
                                    [pose[j, 2] * multi, pose[j_parent, 2] * multi],
                                    [pose[j, 1] * multi, pose[j_parent, 1] * multi],
                                    zdir='z', linewidth=1.5)
            
            elif pose.shape[0] == 43:
                for j, pair in enumerate(dir_vec_pairs_expressive):
                    axes[0].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=1.5)
            
            axes[0].set_xlim3d(-0.5, 0.5)
            axes[0].set_ylim3d(0.5, -0.5)
            axes[0].set_zlim3d(0.5, -0.5)
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('z')
            axes[0].set_zlabel('y')
            axes[0].set_title('{} ({}/{})'.format(name, i + 1, len(output)))

    num_frames = output.shape[0]
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # Save video
    video_path = f'{save_path}/temp_{iter_idx}.mp4'
    ani.save(video_path, fps=15, dpi=150)
    plt.close(fig)

    print(f'Video rendering completed, took {time.time() - start:.1f} seconds.')
    return output_poses

def load_model(dataset, device):
    """
    Loads the VQ-VAE model based on the dataset.

    Args:
        dataset (str): Dataset name ('aqgt' or 'expressive').
        device (torch.device): Device for model inference (CPU/GPU).

    Returns:
        LightningVQVAE: Loaded VQ-VAE model.
    """
    if dataset == 'aqgt':
        ckpt_path = "output/Diffusion_aqgt_styled/VQVAE-epoch=239-body.ckpt"
    elif dataset == 'expressive':
        ckpt_path = "output/Diffusion_exp_styled/VQVAE_EXP_STYLED_BODY_2-epoch=102.ckpt"

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt['hyper_parameters']['args'].data_path = 'data/'
    model = LightningVQVAE(ckpt['hyper_parameters']['args'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)

    return model

def get_dir_vec(code_indices, model, device):
    """
    Converts codebook indices into directional vectors using the model.

    Args:
        code_indices (str): Codebook indices as a string.
        model (LightningVQVAE): The VQ-VAE model.
        device (torch.device): Device for inference.

    Returns:
        numpy.ndarray: Generated directional vectors.
    """
    code_indices = np.array(code_indices.split(','), dtype=int).reshape(1, 4)
    code_indices = torch.from_numpy(code_indices).to(device)
    x_out = model.model.forward_decoder(code_indices)
    return x_out.detach().cpu().numpy()

def get_embeddings(code_indices, model, device):
    """
    Converts codebook indices into embeddings using the model.

    Args:
        code_indices (numpy.ndarray): Codebook indices.
        model (LightningVQVAE): The VQ-VAE model.
        device (torch.device): Device for inference.

    Returns:
        numpy.ndarray: Generated embeddings.
    """
    code_indices = torch.from_numpy(code_indices).to(device)
    x_d = model.model.quantizer.dequantize(code_indices)
    return x_d.detach().cpu().numpy()

def load_mean_vec(dataset):
    """
    Loads the mean pose vectors based on the dataset.

    Args:
        dataset (str): Dataset name ('aqgt' or 'expressive').

    Returns:
        numpy.ndarray: Mean directional vectors.
    """
    if dataset == 'aqgt':
        _, mean_dir_vec = pickle.load(open("aqgt_means.p", "rb"))
    elif dataset == 'expressive':
        _, mean_dir_vec = pickle.load(open("expressive_means.p", "rb"))
    return np.array(mean_dir_vec).reshape(-1, 3)

def plot_four_pca(embeddings1, embeddings2, embeddings3, embeddings4, title1='', title2='', title3='', title4='', subtitle='', save_path='pca.png'):
    """
    Plots PCA (Principal Component Analysis) for four sets of embeddings.

    Args:
        embeddings1, embeddings2, embeddings3, embeddings4 (numpy.ndarray): Input embeddings.
        title1, title2, title3, title4 (str): Titles for each set of embeddings.
        subtitle (str): Subtitle for the plot.
        save_path (str): Path to save the plot.
    """
    pca_1 = PCA(n_components=2)
    pca_2 = PCA(n_components=2)
    pca_3 = PCA(n_components=2)
    pca_4 = PCA(n_components=2)

    embeddings1_reduced = pca_1.fit_transform(embeddings1)
    embeddings2_reduced = pca_2.fit_transform(embeddings2)
    embeddings3_reduced = pca_3.fit_transform(embeddings3)
    embeddings4_reduced = pca_4.fit_transform(embeddings4)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings1_reduced[:, 0], embeddings1_reduced[:, 1], s=5, alpha=0.5, label=title1, color='g')
    plt.scatter(embeddings2_reduced[:, 0], embeddings2_reduced[:, 1], s=5, alpha=0.5, label=title2, color='r')
    plt.scatter(embeddings3_reduced[:, 0], embeddings3_reduced[:, 1], s=5, alpha=0.5, label=title3, color='b')
    plt.scatter(embeddings4_reduced[:, 0], embeddings4_reduced[:, 1], s=5, alpha=0.5, label=title4, color='c')
    plt.title(subtitle)
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300)

def plot_four_tsne(embeddings1, embeddings2, embeddings3, embeddings4, label1='', label2='', label3='', label4='', subtitle='', save_path='tsne.png'):
    """
    Plots t-SNE (t-Distributed Stochastic Neighbor Embedding) for four sets of embeddings.

    Args:
        embeddings1, embeddings2, embeddings3, embeddings4 (numpy.ndarray): Input embeddings.
        label1, label2, label3, label4 (str): Labels for each set of embeddings.
        subtitle (str): Subtitle for the plot.
        save_path (str): Path to save the plot.
    """
    common_dim = 50
    pca = PCA(n_components=common_dim)

    embeddings1_reduced = pca.fit_transform(embeddings1)
    embeddings2_reduced = pca.fit_transform(embeddings2)
    embeddings3_reduced = pca.fit_transform(embeddings3)
    embeddings4_reduced = pca.fit_transform(embeddings4)

    combined_embeddings = np.vstack((embeddings1_reduced, embeddings2_reduced, embeddings3_reduced, embeddings4_reduced))
    tsne = TSNE(n_components=2, random_state=42)
    combined_embeddings_2d = tsne.fit_transform(combined_embeddings)

    n1, n2, n3 = len(embeddings1), len(embeddings2), len(embeddings3)
    embeddings1_2d = combined_embeddings_2d[:n1]
    embeddings2_2d = combined_embeddings_2d[n1:n1+n2]
    embeddings3_2d = combined_embeddings_2d[n1+n2:n1+n2+n3]
    embeddings4_2d = combined_embeddings_2d[n1+n2+n3:]

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], s=5, alpha=0.7, label=label1, color='g')
    plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], s=5, alpha=0.7, label=label2, color='r')
    plt.scatter(embeddings3_2d[:, 0], embeddings3_2d[:, 1], s=5, alpha=0.7, label=label3, color='b')
    plt.scatter(embeddings4_2d[:, 0], embeddings4_2d[:, 1], s=5, alpha=0.7, label=label4, color='c')
    plt.title(subtitle)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300)

def main(dataset):
    """
    Main function to analyze and visualize codebook indices for a specific dataset.

    Args:
        dataset (str): Dataset name ('aqgt' or 'expressive').
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    out_dir = f'output/codebook_default/hand_mu/{dataset}'
    os.makedirs(out_dir, exist_ok=True)

    if dataset == 'aqgt':
        code_indices = pickle.load(open("output/codebook_default/code_indices_hand_mu.p", "rb"))
    elif dataset == 'expressive':
        code_indices = pickle.load(open("output/Diffusion_exp_styled/code_indices_exp.p", "rb"))

    code_indices = np.array(code_indices).reshape(-1, 4).astype(int)

    topMixtures = top_mixtures(code_indices, 10, out_dir)
    topIndices = top_indices(code_indices, 10, out_dir)
    
    plot_histogram(code_indices, dataset, out_dir)
    plot_histogram_of_audio(code_indices, dataset, out_dir)
    plot_histogram_of_text(code_indices, dataset, out_dir)
    plot_histogram_of_style(code_indices, dataset, out_dir)

    model = load_model(dataset, device)
    embeddings = get_embeddings(code_indices, model, device)
    text_embeddings = embeddings[:, 0, :]
    audio_embeddings = embeddings[:, 1, :]
    style_embeddings1 = embeddings[:, 2, :]
    style_embeddings2 = embeddings[:, 3, :]

    plot_four_pca(text_embeddings, audio_embeddings, style_embeddings1, style_embeddings2,
                  'Text aligned embeddings', 'Audio aligned embeddings', 'Style aligned embeddings 1', 'Style aligned embeddings 2',
                  'Distribution of codebook indices based on their modality alignment', os.path.join(out_dir, 'pca.png'))

    plot_four_tsne(text_embeddings, audio_embeddings, style_embeddings1, style_embeddings2,
                   'Text aligned embeddings', 'Audio aligned embeddings', 'Style aligned embeddings 1', 'Style aligned embeddings 2',
                   'Distribution of codebook indices based on their modality alignment', os.path.join(out_dir, 'tsne.png'))

    mean_data = load_mean_vec(dataset)
    for i in range(10):
        print(i)
        dir_vec = get_dir_vec(topMixtures[i][0], model, device)
        create_video_and_save(out_dir, i, 'codebook', dir_vec, mean_data, topMixtures[i][0])

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset != 'aqgt' and dataset != 'expressive':
        print("This dataset is not supported!")
    main(dataset)
