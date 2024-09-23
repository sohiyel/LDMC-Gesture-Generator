"""
Data Preprocessor for creating data samples from video clips, resampling, and normalizing skeleton, audio, and text data.

This script processes data for both AQGT and TED Expressive datasets by resampling the poses, filtering skeletons, 
and generating corresponding audio and text embeddings.
"""

import logging
from collections import defaultdict
import lmdb
import math
import numpy as np
import pyarrow
import tqdm
import utils.data_utils
from data_loader.motion_preprocessor import MotionPreprocessor
from transformers import AutoTokenizer
from models.data2vec.audio_model import AudioEncoder
from models.data2vec.text_model import TextEncoder
import torch
import gc

class DataPreprocessor:
    """
    Class to preprocess video clip data by sampling, resampling skeleton data, 
    generating spectrograms, and obtaining audio/text embeddings.
    
    Attributes:
        dataset_name (str): Name of the dataset ('aqgt' or 'TED_Expressive').
        n_poses (int): Number of poses per data sample.
        subdivision_stride (int): Stride for subdividing poses in the dataset.
        skeleton_resampling_fps (int): Frame rate for skeleton resampling.
        mean_pose (np.array): Mean pose for skeleton normalization.
        mean_dir_vec (np.array): Mean directional vector for skeleton normalization.
        disable_filtering (bool): Whether to disable motion filtering or not.
    """
    
    def __init__(self, dataset_name, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride,
                 pose_resampling_fps, mean_pose, mean_dir_vec, disable_filtering=False):
        """
        Initializes the data preprocessor with necessary configurations.

        Args:
            dataset_name (str): Name of the dataset ('aqgt' or 'TED_Expressive').
            clip_lmdb_dir (str): Path to the source LMDB directory with video clips.
            out_lmdb_dir (str): Path to the output LMDB directory to store processed samples.
            n_poses (int): Number of poses in each sample.
            subdivision_stride (int): Stride for subdividing poses.
            pose_resampling_fps (float): Frame rate for resampling skeletons.
            mean_pose (np.array): Mean pose for skeleton normalization.
            mean_dir_vec (np.array): Mean directional vector for skeleton normalization.
            disable_filtering (bool): If set to True, motion filtering is disabled.
        """
        self.dataset_name = dataset_name
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_pose = mean_pose
        self.mean_dir_vec = mean_dir_vec
        self.disable_filtering = disable_filtering

        # Open the source LMDB environment
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']  # Count the number of videos in the dataset

        # Calculate sample lengths for audio and spectrogram
        self.spectrogram_sample_length = utils.data_utils.calc_spectrogram_length_from_motion_length(self.n_poses, self.skeleton_resampling_fps)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # Initialize tokenizers and encoders
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
        self.audioEncoder = AudioEncoder().cuda()
        self.textEncoder = TextEncoder().cuda()
        self.video2index, self.index2video = {}, {}
        self.nr_videos = 0

        # Create output LMDB environment
        map_size = 1024 * 850  # 850 MB size
        map_size <<= 20  # Convert to bytes
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def run(self, nb_samples=None, clip_duration=None, verbose=True):
        """
        Processes video clips by sampling and normalizing data, saving results in LMDB.
        
        Args:
            nb_samples (int, optional): Maximum number of samples to process.
            clip_duration (tuple, optional): Minimum and maximum duration for clips to consider.
            verbose (bool): Whether to print processing statistics.

        Returns:
        """
        n_filtered_out = defaultdict(int)
        src_txn = self.src_lmdb_env.begin(write=False)
        filtered_result = defaultdict(int)
        cursor = src_txn.cursor()
        counter = 0

        # Iterate over each video clip in the dataset
        for key, value in tqdm.tqdm(cursor):
            try:
                video = pyarrow.deserialize(value)
                vid = video['vid']
                clips = video['clips']
                for clip_idx, clip in enumerate(clips):
                    if clip_duration:
                        clip_t = clip['end_time'] - clip['start_time']
                        if clip_t < clip_duration[0] or clip_t > clip_duration[1]:
                            filtered_result['duration'] += 1
                            continue
                    filtered_result = self._sample_from_clip(vid, clip)
                    counter += 1
                    if nb_samples and counter >= nb_samples:
                        break
                    for type in filtered_result.keys():
                        n_filtered_out[type] += filtered_result[type]
                if nb_samples and counter >= nb_samples:
                    break
            except Exception as e:
                logging.error(e)
                logging.error('Invalid flatbuffers message')

        # Log filtering statistics
        if verbose and not self.disable_filtering:
            with self.dst_lmdb_env.begin() as txn:
                logging.info('no. of samples: {}'.format(txn.stat()['entries']))
                n_total_filtered = sum(n_filtered_out.values())
                logging.info(f'no. of excluded samples: {n_total_filtered} ({100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered):.1f}%)')

        # Close the LMDB environments
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def index_video(self, video):
        if video not in self.video2index:
            self.video2index[video] = self.nr_videos
            self.index2video[self.nr_videos] = video
            self.nr_videos += 1
            return self.nr_videos - 1
        else:
            return self.video2index[video]

    def _sample_from_clip(self, vid, clip):
        """
        Processes a single video clip by resampling skeleton data, generating spectrogram, 
        and tokenizing text for each clip.
        
        Args:
            vid (str): Video ID.
            clip (dict): Clip data including skeletons, audio, and text.

        Returns:
            filtered_out (dict): A dictionary of filtering statistics.
        """
        clip_skeleton = clip['skeletons_3d']
        clip_audio = clip['audio_feat'].reshape((clip['audio_feat'].shape[0], -1))
        clip_audio_raw = clip['audio_raw']
        clip_word_list = clip['words']
        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

        n_filtered_out = defaultdict(int)

        # Resample the skeleton data
        clip_skeleton = utils.data_utils.resample_pose_seq(clip_skeleton, clip_e_t - clip_s_t, self.skeleton_resampling_fps)

        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []
        sample_audio_list = []
        sample_spectrogram_list = []
        sample_words_tokens = []
        sample_video_indices = []

        num_subdivision = math.floor((len(clip_skeleton) - self.n_poses) / self.subdivision_stride) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            # Subdivide skeleton data
            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            if self.dataset_name == "aqgt":
                subdivision_start_time = clip_s_f + start_idx 
                subdivision_end_time = clip_s_f + fin_idx
            elif self.dataset_name == "TED_Expressive":
                subdivision_start_time = clip_s_t + start_idx / self.skeleton_resampling_fps
                subdivision_end_time = clip_s_t + fin_idx / self.skeleton_resampling_fps

            # Get corresponding words for the subdivision
            sample_words = self.get_words_in_time_range(clip_word_list, subdivision_start_time, subdivision_end_time)

            # Resample audio and spectrogram
            audio_start = math.floor(start_idx / len(clip_skeleton) * clip_audio.shape[1])
            audio_end = audio_start + self.spectrogram_sample_length
            if audio_end > clip_audio.shape[1]:
                padded_data = np.pad(clip_audio, ((0, 0), (0, audio_end - clip_audio.shape[1])), mode='symmetric')
                sample_spectrogram = padded_data[:, audio_start:audio_end]
            else:
                sample_spectrogram = clip_audio[:, audio_start:audio_end]

            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            if audio_end > len(clip_audio_raw):
                padded_data = np.pad(clip_audio_raw, (0, audio_end - len(clip_audio_raw)), mode='symmetric')
                sample_audio = padded_data[audio_start:audio_end]
            else:
                sample_audio = clip_audio_raw[audio_start:audio_end]

            if len(sample_words) >= 2:
                texts = ' '.join([word[0] for word in sample_words])
                tokens = self.tokenizer(texts).input_ids
                extended_word_tokens = np.zeros(self.n_poses, dtype=int)
                extended_word_tokens[:min(len(tokens), self.n_poses)] = tokens[:self.n_poses]

                # Filter motion data
                sample_skeletons, filtering_message = MotionPreprocessor(self.dataset_name, sample_skeletons, self.mean_pose, sample_words).get()
                is_correct_motion = bool(sample_skeletons)

                motion_info = {
                    'vid': vid,
                    'start_frame_no': clip_s_f + start_idx,
                    'end_frame_no': clip_s_f + fin_idx,
                    'start_time': subdivision_start_time,
                    'end_time': subdivision_end_time,
                    'is_correct_motion': is_correct_motion,
                    'filtering_message': filtering_message
                }

                if is_correct_motion or self.disable_filtering:
                    # Store processed data
                    sample_skeletons_list.append(sample_skeletons)
                    sample_words_list.append(sample_words)
                    sample_words_tokens.append(extended_word_tokens)
                    sample_audio_list.append(np.array(sample_audio))
                    sample_spectrogram_list.append(sample_spectrogram)
                    aux_info.append(motion_info)
                    sample_video_indices.append(self.index_video(vid))
                else:
                    n_filtered_out[filtering_message] += 1

        # Prepare samples for storage
        for words, word_tokens, poses, audio, spectrogram, aux, index in zip(sample_words_list, sample_words_tokens, sample_skeletons_list,
                                                                             sample_audio_list, sample_spectrogram_list, aux_info, sample_video_indices):
            poses = np.array(poses)
            dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(self.dataset_name, poses)
            normalized_dir_vec = self.normalize_dir_vec(dir_vec, self.mean_dir_vec)
            audio_embeddings = self.get_audio_embeddings(audio)
            text_embeddings = self.get_text_embeddings(word_tokens)

            # Save the processed data as a sample
            v = [words, word_tokens, poses, normalized_dir_vec, audio, spectrogram, aux, audio_embeddings, text_embeddings, index]
            with self.dst_lmdb_env.begin(write=True) as txn:
                k = '{:010}'.format(self.n_out_samples).encode('ascii')
                p = pyarrow.serialize(v).to_buffer()
                txn.put(k, p)
                self.n_out_samples += 1
                if self.n_out_samples % 100 == 0:
                    gc.collect()

        return n_filtered_out

    @staticmethod
    def normalize_dir_vec(dir_vec, mean_dir_vec):
        """Normalizes directional vectors using the provided mean directional vector."""
        return dir_vec - mean_dir_vec

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        """Filters words based on their time range."""
        return [word for word in word_list if word[1] < end_time and word[2] > start_time]

    def get_audio_embeddings(self, audio):
        """Generates audio embeddings from the provided audio data."""
        audio_tensor = torch.tensor(audio).unsqueeze(0).float()
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()

        with torch.no_grad():
            audio_embeddings = self.audioEncoder.get_embeddings(audio_tensor).cpu().numpy()

        return audio_embeddings

    def get_text_embeddings(self, word_tokens):
        """Generates text embeddings from the provided word tokens."""
        word_tensor = torch.tensor(word_tokens).int().unsqueeze(0)
        if torch.cuda.is_available():
            word_tensor = word_tensor.cuda()

        with torch.no_grad():
            text_embeddings = self.textEncoder.get_embeddings(word_tensor).cpu().numpy()

        return text_embeddings
