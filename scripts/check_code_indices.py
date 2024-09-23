from models.vqvae.vqvae_lightning import LightningVQVAE
from data_loader.data_loader import load_data
import torch
import numpy as np
import pickle

def main(model, train_loader, device):
    """
    Processes the training data using the VQ-VAE model to extract code indices 
    and saves them to a pickle file.

    Args:
        model (LightningVQVAE): The VQ-VAE model used for extracting code indices.
        train_loader (DataLoader): DataLoader containing the training data.
        device (torch.device): Device for computation (CPU or GPU).
    """
    # Tensor to store all code indices
    total_code_indices = torch.Tensor().to(device)

    # Iterate over batches of data in the training loader
    for iter_idx, data in enumerate(train_loader, 0):
        _, _, _, target_vec, _, _, _, _, _ = data
        gt_motion = target_vec.to(device)

        # Extract code indices from the model
        code_indices = model.model.get_code_indices(gt_motion)

        # Concatenate the current batch's code indices to the total tensor
        total_code_indices = torch.concat((total_code_indices, code_indices), dim=0)

    # Convert code indices tensor to numpy array and reshape for saving
    print(f"Total code indices shape before reshaping: {total_code_indices.shape}")
    total_code_indices = np.array(total_code_indices.cpu()).reshape(-1, 2)
    print(f"Total code indices shape after reshaping: {total_code_indices.shape}")

    # Save the code indices to a pickle file
    with open('code_indices_aqgt.p', 'wb') as f:
        pickle.dump([total_code_indices], f)

if __name__ == "__main__":
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    ckpt_path = "lightning_logs/version_4638/checkpoints/VQVAE_AQGT_BODY_6_epoch=93.ckpt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint keys: {ckpt.keys()}")

    # Load the data using the arguments stored in the checkpoint
    train_loader, val_loader, test_loader = load_data(ckpt['hyper_parameters']['args'])

    # Initialize the VQ-VAE model and load its state from the checkpoint
    model = LightningVQVAE(ckpt['hyper_parameters']['args'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)

    # Call the main function to process the training data and save the code indices
    main(model, train_loader, device)
