from models.diffusion.diffusion_lightning import LightningDiffusion
from models.data2vec.audio_model import AudioEncoder
from models.data2vec.text_model import TextEncoder
from transformers import AutoTokenizer
from data_loader.data_preprocessor import DataPreprocessor
from utils.option_video import get_args_parser
from utils.train_utils import set_logger
from utils.data_utils import convert_dir_vec_to_pose, convert_pose_seq_to_dir_vec, resample_pose_seq, dir_vec_pairs_expressive
import logging
import torch
import pickle
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import subprocess
import soundfile as sf
import datetime
import math
import pyarrow
import lmdb

# Initialize models and tokenizers
tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
audioEncoder = AudioEncoder().cuda()
textEncoder = TextEncoder().cuda()

def create_video_and_save(dataset_name, save_path, iter_idx, prefix, target, output1, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
    """
    Renders a video comparing human poses with model-generated poses, optionally merging with audio.

    Args:
        dataset_name (str): Name of the dataset used (e.g., 'aqgt' or 'TED_Expressive').
        save_path (str): Directory path to save the video.
        iter_idx (int): Index of the iteration for the video filename.
        prefix (str): Prefix for the filename.
        target (np.ndarray): Ground truth poses (optional).
        output1 (np.ndarray): Model-generated poses.
        mean_data (np.ndarray): Mean data for un-normalization.
        title (str): Title for the video.
        audio (np.ndarray, optional): Audio waveform for merging with video.
        aux_str (str, optional): Additional string for the title.
        clipping_to_shortest_stream (bool): Whether to clip audio to the shortest stream.
        delete_audio_file (bool): Whether to delete the intermediate audio file after merging.
    """
    print('Rendering a video...')
    start = time.time()

    # Set up the figure for animation
    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    
    fig_title = title
    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # Un-normalization and conversion to poses
    mean_data = mean_data.flatten()
    if dataset_name == "TED_Expressive":
        output1 = output1.reshape(-1, 126)
    output1 += mean_data
    output_poses1 = convert_dir_vec_to_pose(dataset_name, output1)

    if target is not None:
        target += mean_data
        target_poses = convert_dir_vec_to_pose(dataset_name, target)
    else:
        target_poses = None

    models_list = ['human', 'our']

    def animate(i):
        for k, name in enumerate(models_list):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'our' and i < len(output1):
                pose = output_poses1[i]
            else:
                pose = None

            if pose is not None:
                pose = np.reshape(pose, (-1, 3))
                axes[k].clear()

                # Plot skeleton based on dataset type
                if dataset_name == "aqgt":
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
                        multi = 3
                        axes[k].plot([pose[j, 0] * multi, pose[j_parent, 0] * multi],
                                        [pose[j, 2] * multi, pose[j_parent, 2] * multi],
                                        [pose[j, 1] * multi, pose[j_parent, 1] * multi],
                                        zdir='z', linewidth=1.5)
                elif dataset_name == "TED_Expressive":
                    for pair in dir_vec_pairs_expressive:
                        axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                     [pose[pair[0], 2], pose[pair[1], 2]],
                                     [pose[pair[0], 1], pose[pair[1], 1]], zdir='z', linewidth=1.5)

                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_xlabel('x')
                axes[k].set_ylabel('z')
                axes[k].set_zlabel('y')
                axes[k].set_title(models_list[k])
                axes[k].axis('off')

    num_frames = max(len(target), len(output1)) if target is not None else len(output1)
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # Save the video
    video_path = f'{save_path}/temp_{iter_idx}.mp4'
    ani.save(video_path, fps=15, dpi=300)
    plt.close(fig)

    # Merge with audio
    if audio is not None:
        audio_path = f'{save_path}/{iter_idx}.wav'
        audio = np.float32(audio)
        sf.write(audio_path, audio, 16000)

        merged_video_path = f'{save_path}/{iter_idx}_{models_list[0]}_{models_list[1]}.mp4'
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2', merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(-1, '-shortest')
        subprocess.call(cmd)

        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print(f"Video rendering completed, took {time.time() - start:.1f} seconds.")
    return output_poses1, target_poses

def get_audio_embeddings(audio):
    """
    Extracts audio embeddings using the AudioEncoder.

    Args:
        audio (np.ndarray): Raw audio waveform.

    Returns:
        torch.Tensor: Audio embeddings.
    """
    audio_tensor = torch.tensor(audio).unsqueeze(0).float().cuda()
    with torch.no_grad():
        return audioEncoder.get_embeddings(audio_tensor)

def get_text_embeddings(word_tokens):
    """
    Extracts text embeddings using the TextEncoder.

    Args:
        word_tokens (list): List of tokenized word indices.

    Returns:
        torch.Tensor: Text embeddings.
    """
    word_tensor = torch.tensor(word_tokens).int().unsqueeze(0).cuda()
    with torch.no_grad():
        return textEncoder.get_embeddings(word_tensor)

def generate_gestures(args, models, audio, words, clip_s_f, audio_sr=16000, seed_seq=None, fade_out=False):
    """
    Generates gesture sequences from audio and text embeddings using a model.

    Args:
        args: Arguments containing model configurations.
        models (dict): Dictionary containing body and hand models.
        audio (np.ndarray): Raw audio waveform.
        words (list): List of word tokens.
        clip_s_f (float): Start frame for the clip.
        audio_sr (int): Sample rate of the audio.
        seed_seq (torch.Tensor, optional): Seed sequence for initial poses.
        fade_out (bool): Whether to fade out the gestures to the mean pose.

    Returns:
        np.ndarray: Generated gesture sequences.
    """
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr
    pre_seq = torch.zeros((4, 159)) if args.dataset_name == "aqgt" else torch.zeros((4, 126))
    if seed_seq is not None:
        pre_seq = torch.Tensor(seed_seq)

    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    num_subdivision = 1 if clip_length < unit_time else math.ceil((clip_length - unit_time) / stride_time) + 1
    audio_sample_length = int(unit_time * audio_sr)

    start = time.time()
    for i in range(num_subdivision):
        start_time = i * stride_time
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        in_audio = np.pad(audio[audio_start:audio_end], (0, audio_sample_length - len(in_audio)), 'constant')
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        # Prepare text embeddings
        subdivision_start_time = clip_s_f + i * 30
        subdivision_end_time = subdivision_start_time + 34
        word_seq = DataPreprocessor.get_words_in_time_range(words, subdivision_start_time, subdivision_end_time)
        tokens = tokenizer(' '.join([word[0] for word in word_seq])).input_ids[:34]
        extended_word_tokens = np.zeros(34, dtype=int)
        extended_word_tokens[:len(tokens)] = tokens

        # Generate embeddings
        audio_embeddings = get_audio_embeddings(audio)
        text_embeddings = get_text_embeddings(extended_word_tokens)
        pre_seq = pre_seq.float().to(device)

        # Generate poses
        pred_body = models['body'](audio_embeddings, text_embeddings, pre_seq)
        pred_hand = models['hand'](audio_embeddings, text_embeddings, pre_seq)
        if args.dataset_name == "aqgt":
            out_dir_vec = torch.concat([pred_body, pred_hand],dim=-1)
        elif args.dataset_name == "TED_Expressive":
            body_indices = [0,1,2,3,4,20,21,37,38,39,40,41]
            mask = np.zeros(42, dtype=bool)
            mask[body_indices] = True
            out_dir_vec = torch.zeros(1, 34, 42, 3).to(pred_body.device)
            out_dir_vec[:, :, mask, :] = pred_body.reshape(pred_body.shape[0], pred_body.shape[1], -1, 3)
            out_dir_vec[:, :, ~mask, :] = pred_hand.reshape(pred_hand.shape[0], pred_hand.shape[1], -1, 3)

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # Smooth motion transitions
        if out_list:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    out_dir_vec = np.vstack(out_list)

    # Fade out to mean pose if required
    if fade_out:
        fade_out_sequence(out_dir_vec, args)

    return out_dir_vec

def fade_out_sequence(out_dir_vec, args):
    """
    Smoothly fades out the generated sequence to the mean pose.

    Args:
        out_dir_vec (np.ndarray): Generated sequence of direction vectors.
        args: Arguments containing mean direction vector and other configurations.
    """
    n_smooth = args.n_pre_poses
    start_frame = len(out_dir_vec) - n_smooth * 2
    end_frame = start_frame + n_smooth * 2
    out_dir_vec[end_frame-n_smooth:] = np.zeros((len(args.mean_dir_vec)))

    # Interpolation for smoothing
    y = out_dir_vec[start_frame:end_frame]
    x = np.array(range(0, y.shape[0]))
    w = np.ones(len(y))
    w[0], w[-1] = 5, 5
    coeffs = np.polyfit(x, y, 2, w=w)
    interpolated_y = np.transpose(np.array([np.poly1d(coeffs[:, k])(x) for k in range(0, y.shape[1])]))
    out_dir_vec[start_frame:end_frame] = interpolated_y

if __name__ == "__main__":
    args = get_args_parser()
    set_logger(args.out_dir, os.path.basename(__file__).replace('.py', '.log'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load models and mean direction vector
    models = {}
    logging.info(f"Loading diffusion body model from {args.body_diffusion_path}")
    ckpt_body = torch.load(args.body_diffusion_path, map_location='cpu')
    ckpt_body["hyper_parameters"]["args"].data_path = args.data_path
    ckpt_body["hyper_parameters"]["args"].vqvae_path = args.body_vqvae_path
    model_body = LightningDiffusion(ckpt_body["hyper_parameters"]["args"])
    model_body.load_state_dict(ckpt_body['state_dict'])
    model_body.eval().to(device)
    models['body'] = model_body

    logging.info(f"Loading diffusion hand model from {args.hand_diffusion_path}")
    ckpt_hand = torch.load(args.hand_diffusion_path, map_location='cpu')
    ckpt_hand["hyper_parameters"]["args"].data_path = args.data_path
    ckpt_hand["hyper_parameters"]["args"].vqvae_path = args.hand_vqvae_path
    model_hand = LightningDiffusion(ckpt_hand["hyper_parameters"]["args"])
    model_hand.load_state_dict(ckpt_hand['state_dict'])
    model_hand.eval().to(device)
    models['hand'] = model_hand

    logging.info(f"Loading mean direction vector from {args.mean_vectors_path}")
    mean_pose, mean_dir_vec = pickle.load(open(args.mean_vectors_path, "rb"))
    mean_dir_vec = np.array(mean_dir_vec)
    mean_dir_vec = mean_dir_vec.reshape(-1, 159) if args.dataset_name == "aqgt" else mean_dir_vec.reshape(-1, 126)
    args.mean_dir_vec = mean_dir_vec
    clip_duration_range = args.clip_duration
    n_generations = args.nb_videos

    # load clips and make gestures
    n_saved = 0

    lmdb_env = lmdb.open(os.path.join(args.data_path, args.dataset, args.test_data_path), readonly=True, lock=False)

    with lmdb_env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
        while n_saved < n_generations:  # loop until we get the desired number of results
            # select video
            key = random.choice(keys)

            buf = txn.get(key)
            video = pyarrow.deserialize(buf)
            vid = video['vid']
            clips = video['clips']
            n_clips = len(clips)
            if n_clips == 0:
                continue
            clip_idx = random.randrange(n_clips)

            clip_poses = clips[clip_idx]['skeletons_3d']
            clip_audio = clips[clip_idx]['audio_raw']
            clip_audio = np.array(clip_audio)
            clip_words = clips[clip_idx]['words']
            clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]
            clip_s_f, clip_e_f = clips[clip_idx]['start_frame_no'], clips[clip_idx]['end_frame_no']

            clip_poses = resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                            args.motion_resampling_framerate)
            target_dir_vec = convert_pose_seq_to_dir_vec(args.dataset_name, clip_poses)
            target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
            
            target_dir_vec -= mean_dir_vec

            # check duration
            clip_duration = clip_time[1] - clip_time[0]
            if clip_duration < clip_duration_range[0] or clip_duration > clip_duration_range[1]:
                continue

            # synthesize
            for selected_vi in range(len(clip_words)):  # make start time of input text zero
                clip_words[selected_vi][1] -= clip_time[0]  # start time
                clip_words[selected_vi][2] -= clip_time[0]  # end time

            out_dir_vec = generate_gestures(args, models, clip_audio, clip_words, clip_s_f, 
                                            seed_seq=target_dir_vec[0:args.n_pre_poses], fade_out=False)

            # make a video
            aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                    str(datetime.timedelta(seconds=clip_time[1])))
            mean_data = np.array(mean_dir_vec).reshape(-1, 3)
            save_path = args.out_dir
            create_video_and_save(
                args.dataset_name,
                save_path, n_saved, 'long',
                target_dir_vec, out_dir_vec, mean_data,
                '', audio=clip_audio, aux_str=aux_str)
            n_saved += 1