from models.vqvae.vqvae_lightning import LightningVQVAE
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.option_vq_2 import get_args_parser
from utils.train_utils import set_logger
from data_loader.data_loader import load_data
from lightning import Trainer
import pprint
import logging
import torch
import os

def main(args):
    """
    Main function for training the VQ-VAE model using PyTorch Lightning.
    
    Args:
        args: Parsed arguments containing model and training configurations.
    """
    # If data_path is TMPDIR, replace it with the actual environment variable path
    if args.data_path == 'TMPDIR':
        args.data_path = os.environ[args.data_path]

    # Set up logging
    set_logger(args.out_dir, os.path.basename(__file__).replace('.py', '.log'))
    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs detected".format(torch.cuda.device_count()))
    logging.info(pprint.pformat(vars(args)))

    # Load training and validation datasets
    train_loader, val_loader, _ = load_data(args)

    # Initialize the VQ-VAE model with the provided arguments
    model = LightningVQVAE(args)

    # Set up checkpoint callback for saving model checkpoints
    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k,  # Save the top k models
        monitor=args.monitor,        # Metric to monitor for checkpointing
        mode=args.mode,              # Minimize or maximize the monitored metric
        dirpath=os.path.join(args.out_dir, args.exp_name),  # Path to save checkpoints
        filename="VQVAE-{epoch:02d}"  # Checkpoint filename format
    )

    # Initialize the Trainer with the appropriate settings
    trainer = Trainer(
        gpus=args.num_gpu,  # Number of GPUs to use for training
        max_epochs=args.epochs,  # Maximum number of epochs
        check_val_every_n_epoch=1,  # Validate after every epoch
        strategy=DDPStrategy(find_unused_parameters=True),  # Use Distributed Data Parallel (DDP)
        enable_checkpointing=True,  # Enable checkpoint saving
        callbacks=[checkpoint_callback]  # Pass the checkpoint callback to the trainer
    )

    # If resuming from a checkpoint, continue training from the checkpoint
    if args.resume_path:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume_path  # Path to the checkpoint for resuming training
        )
    else:
        # Start training a new model from scratch
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

if __name__ == "__main__":
    # Parse arguments and start the training process
    args = get_args_parser()
    main(args=args)
