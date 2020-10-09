"""
Runs a model on a single node across multiple gpus.
"""
import os
import numpy as np
import warnings
import random
import torch

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

from project.model.deepspeech_main import DeepSpeech

#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
seed_everything(seed)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = DeepSpeech(**vars(args))
    # ------------------------
    # 2 INIT LOGGERS and special CALLBACKS
    # ------------------------
    # Early stopper
    early_stop = EarlyStopping(
        monitor=args.early_stop_metric,
        patience=args.early_stop_patience,
        verbose=True,
    )
    # Checkpoint manager
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=5,  # Save 5 Top models
        monitor="wer",
        mode="min",
        period=1,
    )
    # Loggers
    logger = TensorBoardLogger(save_dir=args.logs_path, name=args.experiment_name)
    lr_logger = LearningRateLogger(logging_interval='step')
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
                    gradient_clip_val=0,
                    auto_scale_batch_size=False,
                    gpus=0,
                    auto_select_gpus=True,
                    log_gpu_memory=True,
                    # precision=args.precision,
                    logger=logger,
                    row_log_interval = args.batch_size,
                    early_stop_callback = early_stop,
                    checkpoint_callback= checkpoint_callback,
                    callbacks=[lr_logger],
                    fast_dev_run=False,
                    # resume_from_checkpoint='/mnt/data/github/DeepSpeech-pytorch/runs/DeepSpeech_onecycle_defaultbits/version_1/checkpoints/epoch=12.ckpt',
                    # auto_lr_find='learning_rate',
                    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(model)

def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    #Model parser
    parser = DeepSpeech.add_model_specific_args(parent_parser)
    # Data
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--data_root", default="/mnt/data/github/DeepSpeech-pytorch/data/", type=str)
    parser.add_argument("--data_train", default=["train-clean-100", "train-clean-360", "train-other-500"])
    parser.add_argument("--data_test", default=["test-clean"])
    # Training params (opt)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    #parser.add_argument("--precission", default=16, type=int)
    parser.add_argument("--early_stop_metric", default="wer", type=str)
    parser.add_argument("--logs_path", default="runs/", type=str)
    parser.add_argument("--experiment_name", default="DeepSpeech", type=str)
    parser.add_argument("--early_stop_patience", default=3, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    #Precission args
    parser.add_argument("--amp_level", default='02', type=str)
    parser.add_argument("--precision", default=32, type=int)

    args = parser.parse_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
