from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import SpeechMotionDataset, default_collate_fn
import os
import pickle
import numpy as np

def load_data(args):
    """
    Loads and prepares the training, validation, and test datasets.

    Args:
        args: An object containing the necessary arguments, such as:
            - dataset_name: The name of the dataset to be used (e.g., 'aqgt' or 'TED_Expressive').
            - data_path: The root path to the dataset directory.
            - train_data_path: The path to the training data within the dataset directory.
            - val_data_path: The path to the validation data within the dataset directory.
            - test_data_path: The path to the test data within the dataset directory.
            - n_poses: Number of poses to use in each data sample.
            - subdivision_stride: The stride for dividing the poses in the dataset.
            - motion_resampling_framerate: The frame rate to resample the motion data.
            - batch_size: The number of samples per batch for DataLoader.
            - loader_workers: The number of worker threads for data loading.

    Returns:
        A tuple containing:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
            - test_loader: DataLoader for the test dataset.
    """
    
    # Set the appropriate collate function for data batching
    collate_fn = default_collate_fn

    # Load the mean pose and directional vector based on the dataset type
    if args.dataset_name == "aqgt":
        mean_pose, mean_dir_vec = pickle.load(open("aqgt_means.p", "rb"))
    elif args.dataset_name == "TED_Expressive":
        mean_pose, mean_dir_vec = pickle.load(open("expressive_means.p", "rb"))
    mean_pose, mean_dir_vec = np.array(mean_pose), np.array(mean_dir_vec)

    # Initialize the training dataset
    train_dataset = SpeechMotionDataset(
        dataset_name=args.dataset_name,  # Specify which dataset is being used
        lmdb_dir=os.path.join(args.data_path, args.dataset, args.train_data_path),
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        mean_dir_vec=mean_dir_vec,
        mean_pose=mean_pose,
        remove_word_timing=True  # Remove word timing information (specific to the dataset)
    )

    # Create the DataLoader for training data
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle data to improve training randomness
        drop_last=True,  # Drop the last incomplete batch
        num_workers=args.loader_workers,
        pin_memory=True,  # Pin memory to improve transfer to GPU (if applicable)
        collate_fn=collate_fn  # Use the custom collate function
    )

    # Initialize the validation dataset
    val_dataset = SpeechMotionDataset(
        dataset_name=args.dataset_name,
        lmdb_dir=os.path.join(args.data_path, args.dataset, args.val_data_path),
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        mean_dir_vec=mean_dir_vec,
        mean_pose=mean_pose,
        remove_word_timing=True  # Remove word timing information
    )

    # Create the DataLoader for validation data
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for validation
        drop_last=True,  # Drop the last incomplete batch
        num_workers=args.loader_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize the test dataset
    test_dataset = SpeechMotionDataset(
        dataset_name=args.dataset_name,
        lmdb_dir=os.path.join(args.data_path, args.dataset, args.test_data_path),
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        mean_dir_vec=mean_dir_vec,
        mean_pose=mean_pose
    )

    # Create the DataLoader for test data
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle test data (can be set to False based on the evaluation strategy)
        drop_last=True,
        num_workers=args.loader_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Return the DataLoaders for training, validation, and testing
    return train_loader, val_loader, test_loader
