from models.diffusion.diffusion_lightning import LightningDiffusion
from lightning import Trainer
from utils.option_diffusion import get_args_parser
from utils.train_utils import set_logger
from data_loader.data_loader import load_data
import pprint
import logging
import torch
import os
from lightning.pytorch.callbacks import ModelCheckpoint

def main(args):
    """
    Main training function for the diffusion model using PyTorch Lightning.
    
    Args:
        args: Parsed arguments containing model configurations and training parameters.
    """
    # Resolve environment variable if needed
    if args.data_path == 'TMPDIR':
        args.data_path = os.environ[args.data_path]

    # Set up logging for the training process
    set_logger(args.out_dir, os.path.basename(__file__).replace('.py', '.log'))
    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(f"{torch.cuda.device_count()} GPUs detected, default settings applied.")
    logging.info(pprint.pformat(vars(args)))

    # Load VQVAE checkpoint and retrieve its arguments
    ckpt = torch.load(args.vqvae_path, map_location='cpu')
    vqvae_args = ckpt['hyper_parameters']['args']

    # Load the training and validation datasets
    train_loader, val_loader, _ = load_data(args)

    # Initialize the diffusion model with the provided arguments
    model = LightningDiffusion(args)

    # Set up the output directory for model checkpoints
    out_dir = os.path.join(args.out_dir, args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,  # Save the top 2 checkpoints
        monitor="FGD",  # Monitor Frechet Gesture Distance (FGD)
        mode="min",  # Minimize the FGD
        dirpath=out_dir,  # Directory to save checkpoints
        filename="Diffusion-{epoch:02d}",  # Filename format for saved checkpoints
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator='gpu',  # Use GPU for training
        max_epochs=args.epochs,  # Maximum number of epochs
        check_val_every_n_epoch=args.eval_iter,  # Frequency of validation checks
        gpus=args.num_gpu,  # Number of GPUs to use
        strategy='ddp',  # Distributed data parallel strategy for multi-GPU
        enable_checkpointing=True,  # Enable checkpointing
        enable_progress_bar=True,  # Disable progress bar (can enable for debugging)
        callbacks=[checkpoint_callback],  # Use the checkpoint callback
    )
    
    # If resuming from a previous checkpoint, continue training from there
    if args.resume_path:
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader, 
                    ckpt_path=args.resume_path)
    else:
        # Start a fresh training session
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader)

if __name__ == "__main__":
    # Parse the arguments and pass them to the main function
    args = get_args_parser()
    main(args=args)
