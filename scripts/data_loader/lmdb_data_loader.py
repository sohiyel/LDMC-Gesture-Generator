import logging
import os
import numpy as np
import lmdb
import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils.data_utils import calc_spectrogram_length_from_motion_length, make_audio_fixed_length
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow
import copy

def default_collate_fn(data):
    """
    Custom collate function for batching data with varying sequence lengths.

    Args:
        data (list): A list of tuples containing word sequences, pose sequences, audio, etc.

    Returns:
        Tuple of batched tensors: words, word_tokens, pose_seq, vec_seq, audio, spectrogram, aux_info, 
        audio_embeddings, text_embeddings, and video_indices.
    """
    # Unzip the data into separate components
    words, word_tokens, pose_seq, vec_seq, audio, spectrogram, aux_info, audio_embeddings, text_embeddings, video_indices = zip(*data)

    # Apply the default collation to each component
    word_tokens = default_collate(word_tokens)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}
    audio_embeddings = default_collate(audio_embeddings)
    text_embeddings = default_collate(text_embeddings)
    video_indices = default_collate(video_indices)

    return words, word_tokens, pose_seq, vec_seq, audio, spectrogram, aux_info, audio_embeddings, text_embeddings, video_indices


class SpeechMotionDataset(Dataset):
    """
    Dataset class to load speech and motion data from LMDB storage.

    Attributes:
        lmdb_dir (str): Path to the LMDB directory containing the data.
        n_poses (int): Number of poses per data sample.
        subdivision_stride (int): Stride for subdividing the poses in the data.
        skeleton_resampling_fps (int): Frame rate for resampling skeleton motion data.
        mean_pose (np.array): Mean pose for normalization.
        mean_dir_vec (np.array): Mean directional vector for normalization.
        remove_word_timing (bool): If True, removes word timing information from the data.
    """
    
    def __init__(self, dataset_name, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec,
                 remove_word_timing=False):
        """
        Initializes the dataset by setting up paths and resampling configurations.

        Args:
            dataset_name (str): The name of the dataset (e.g., 'aqgt' or 'TED_Expressive').
            lmdb_dir (str): The path to the LMDB directory.
            n_poses (int): The number of poses per data sample.
            subdivision_stride (int): The stride to use when subdividing the data.
            pose_resampling_fps (float): The frame rate for resampling motion data.
            mean_pose (np.array): The mean pose used for normalization.
            mean_dir_vec (np.array): The mean directional vector used for normalization.
            remove_word_timing (bool): If True, removes word timing information.
        """
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_dir_vec = mean_dir_vec
        self.remove_word_timing = remove_word_timing

        # Calculate expected lengths for audio and spectrogram data
        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(n_poses, pose_resampling_fps)

        logging.info("Reading data from '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'

        # Create a cache of the dataset if it doesn't already exist
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            assert mean_dir_vec is not None
            if mean_dir_vec.shape[-1] != 3:
                mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
            n_poses_extended = int(round(n_poses * 1.25))  # Add a margin to n_poses for flexibility
            data_sampler = DataPreprocessor(dataset_name, lmdb_dir, preloaded_dir, n_poses_extended,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # Initialize the LMDB environment for reading
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']  # Number of samples in the dataset

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Loads and returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            A tuple containing:
                - word_seq: The word sequence for the sample.
                - word_tokens: Tokenized word sequence.
                - pose_seq: Pose sequence for the sample.
                - vec_seq: Directional vector sequence for the poses.
                - audio: The audio data for the sample.
                - spectrogram: The spectrogram for the audio data.
                - aux_info: Auxiliary information related to the sample.
                - audio_embeddings: Embeddings for the audio data.
                - text_embeddings: Embeddings for the text data.
                - video_indices: Indices of the video segments.
        """
        with self.lmdb_env.begin(write=False) as txn:
            # Retrieve the sample from LMDB using the index
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)

            # Unpack the sample
            word_seq, word_tokens, pose_seq, vec_seq, audio, spectrogram, aux_info, audio_embeddings, text_embeddings, video_indices = sample

        # Perform clipping on the sample based on timing
        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None

        # Convert data to PyTorch tensors
        word_tokens = torch.from_numpy(copy.copy(np.array(word_tokens))).int()
        vec_seq = torch.from_numpy(copy.copy(vec_seq)).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.from_numpy(copy.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(copy.copy(audio)).float()
        spectrogram = torch.from_numpy(copy.copy(spectrogram)).float()
        audio_embeddings = torch.from_numpy(copy.copy(audio_embeddings)).float()
        text_embeddings = torch.from_numpy(copy.copy(text_embeddings)).float()
        video_indices = torch.from_numpy(copy.copy(np.array(video_indices))).int()

        return word_seq, word_tokens, pose_seq, vec_seq, audio, spectrogram, aux_info, audio_embeddings, text_embeddings, video_indices
