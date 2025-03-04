#!/usr/bin/env python3
# @brief:    Test script for range image-based point cloud prediction
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import os
import time
import argparse
import random
import yaml
import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies.ddp import DDPStrategy

from datasets import KittiOdometryModule
from impnet.models.impnet import IMPNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./test.py")
    parser.add_argument(
        "--model", "-m", type=str, default='best_instance', help="Model to be tested"
    )
    parser.add_argument("--last", action="store_true", help="Use last saved model")
    parser.add_argument(
        "--limit_test_batches",
        "-l",
        type=float,
        default=1.0,
        help="Percentage of test data to be tested",
    )
    parser.add_argument("--save", "-s", action="store_true",default=True, help="Save point clouds")
    parser.add_argument(
        "--cd_downsample",
        type=int,
        default=-1,
        help="Number of downsampled points for evaluating Chamfer Distance",
    )
    parser.add_argument("--path", "-p", type=str, default=None, help="Path to data")
    parser.add_argument(
        "-seq",
        "--sequence",
        type=int,
        nargs="+",
        default=None,
        help="Sequence to be tested",
    )

    args, unparsed = parser.parse_known_args()
    # load config file
    config_filename = "./impnet/runs/" + args.model + "/hparams.yaml"
    cfg = yaml.safe_load(open(config_filename))
    print("cfg",config_filename)
    print("save",cfg["TEST"]["SAVE_POINT_CLOUDS"])
    print("Starting testing model ", cfg["LOG_NAME"])
    """Manually set these"""
    cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"] = False
    cfg["DATA_CONFIG"]["GENERATE_FILES"] = False
    cfg["TEST"]["SAVE_POINT_CLOUDS"] = args.save
    cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"] = args.cd_downsample
    print("Evaluating CD on ", cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"], " points.")

    if args.sequence:
        cfg["DATA_CONFIG"]["SPLIT"]["TEST"] = args.sequence
        cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"] = args.sequence
        cfg["DATA_CONFIG"]["SPLIT"]["VAL"] = args.sequence

    #seed = time.time()
    #random.seed(seed)
    #cfg["SEED"] = seed
    #print("Random seed is ", cfg["SEED"])

    ###### Dataset
    data = KittiOdometryModule(cfg)
    print("data object created")
    data.setup()
    print("data setup done")

    ###### Load checkpoint
    if args.last:
        checkpoint_path = "./impnet/runs/" + args.model + "/checkpoints/last.ckpt"
    else:
        checkpoint_path = "./impnet/runs/" + args.model + "/checkpoints/min_val_loss.ckpt"
    cfg["TEST"]["USED_CHECKPOINT"] = checkpoint_path
    print(checkpoint_path)
    ###### Model
    model = IMPNet(cfg, num_channels=1, num_kernels=32, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    img_size=(64, 64), num_layers=3, peep=False)


    # Only log if test is done on full data
    if args.limit_test_batches == 1.0:
        logger = TensorBoardLogger(
            save_dir="./impnet/runs/" + args.model, default_hp_metric=False, name="test", version=""
        )
    else:
        logger = False

    ###### Trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        strategy = DDPStrategy(find_unused_parameters=False),
        limit_test_batches=1.0
    )

    print("Starting TEST...")
    ###### Testing
    torch.set_float32_matmul_precision('high') 
    results = trainer.test(model, data.test_dataloader(), ckpt_path=checkpoint_path)

    if True:
        filename = os.path.join(
            cfg["LOG_DIR"], "test", "results_" + time.strftime("%Y%m%d_%H%M%S") + ".yml"
        )
        log_to_save = {**{"results": results}, **vars(args), **cfg}
        with open(filename, "w") as yaml_file:
            yaml.dump(log_to_save, yaml_file, default_flow_style=False)
