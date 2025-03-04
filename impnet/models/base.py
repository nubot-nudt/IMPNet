#!/usr/bin/env python3
# @brief:    Point cloud prediction architecture with Conv-LSTM and spatial and channel-wise attention
# @author    Kaustab Pal  [kaustab21@gmail.com]]
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from impnet.models.loss import Loss
from impnet.utils.projection import projection
from impnet.utils.logger import log_point_clouds, save_range_and_mask, save_point_clouds
from impnet.models.cluster import cluster_instance,gen_color_map
import open3d as o3d
class BasePredictionModel(pl.LightningModule):
    """Pytorch Lightning base model for point cloud prediction"""

    def __init__(self, cfg):
        """Init base model

        Args:
            cfg (dict): Config parameters
        """
        super(BasePredictionModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.min_range = self.cfg["DATA_CONFIG"]["MIN_RANGE"]
        self.max_range = self.cfg["DATA_CONFIG"]["MAX_RANGE"]
        self.register_buffer("mean", torch.Tensor(self.cfg["DATA_CONFIG"]["MEAN"]))
        self.register_buffer("std", torch.Tensor(self.cfg["DATA_CONFIG"]["STD"]))
        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.use_xyz = self.cfg["MODEL"]["USE"]["XYZ"]
        self.use_intensity = self.cfg["MODEL"]["USE"]["INTENSITY"]

        # Create list of index used in input
        self.inputs = [0]
        if self.use_xyz:
            self.inputs.append(1)
            self.inputs.append(2)
            self.inputs.append(3)
        if self.use_intensity:
            self.inputs.append(4)
        self.n_inputs = len(self.inputs)

        # Init loss
        self.loss = Loss(self.cfg)

        # Init projection class for re-projcecting from range images to 3D point clouds
        self.projection = projection(self.cfg)

        self.chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
        self.instance = cluster_instance(self.cfg)

    def forward(self, x,x_res):
        pass

    def configure_optimizers(self):
        """Optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg["TRAIN"]["LR_EPOCH"],
            gamma=self.cfg["TRAIN"]["LR_DECAY"],
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step including logging

        Args:
            batch (dict): A dict with a batch of training samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        past = batch["past_data"]
        x_res = batch["res_data"]
        future = batch["fut_data"]
        mos_label = batch["mos_label"]
        instance_label = batch["instance_label"]
        output, _ = self.forward(past,x_res)
        batch_size, n_inputs, n_future_steps, H, W = past.shape
        loss = self.loss(output, future,mos_label,instance_label, "train", self.current_epoch)

        
        self.log("train/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)
        self.log("train/mean_chamfer_distance", loss["mean_chamfer_distance"], sync_dist=True)
        self.log("train/final_chamfer_distance", loss["final_chamfer_distance"], sync_dist=True)
        self.log("train/loss_range_view", loss["loss_range_view"], sync_dist=True)
        self.log("train/loss_mask", loss["loss_mask"], sync_dist=True)
        self.log("train/loss_mos", loss["loss_mos"], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning validation step including logging

        Args:
            batch (dict): A dict with a batch of validation samples
            batch_idx (int): Index of batch in dataset

        Returns:
            None
        """
        past = batch["past_data"]
        x_res = batch["res_data"]
        future = batch["fut_data"]
        mos_label = batch["mos_label"]
        instance_label = batch["instance_label"]
        output, _ = self.forward(past,x_res)
        loss = self.loss(output, future, mos_label ,instance_label,"val", self.current_epoch)
        #self.instance.get_clustered_point_id(output)
        monitor_loss = loss["mean_chamfer_distance"] + loss["loss_range_view"]\
                + loss["loss_mask"]+loss["loss_mos"]
        self.log("val/loss", monitor_loss, sync_dist=True, on_epoch=True)
        self.log(
            "val/mean_chamfer_distance", loss["mean_chamfer_distance"], sync_dist=True, prog_bar=True, on_epoch=True
        )
        self.log(
            "val/final_chamfer_distance", loss["final_chamfer_distance"], on_epoch=True
        )
        self.log("val/loss_range_view", loss["loss_range_view"], sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("val/loss_mask", loss["loss_mask"], on_epoch=True)
        self.log("val/loss_mos", loss["loss_mos"], on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        past = batch["past_data"]
        future = batch["fut_data"]
        x_res = batch["res_data"]
        mos_label = batch["mos_label"]
        instance_label =batch["instance_label"]
        batch_size, n_inputs, n_future_steps, H, W = past.shape

        start = time.time()
        output, attn = self.forward(past,x_res)
        inference_time = (time.time() - start) / batch_size
        self.log("test/inference_time", inference_time, on_epoch=True)
        
        loss = self.loss(output, future, mos_label,instance_label,"test", self.current_epoch)

        self.log("test/loss_range_view", loss["loss_range_view"], on_epoch=True)
        self.log("test/loss_mask", loss["loss_mask"], on_epoch=True)
        #self.log("test/loss_mos", loss["loss_mos"], on_epoch=True)

        for step, value in loss["chamfer_distance"].items():
            self.log("test/chamfer_distance_{:d}".format(step), value, on_epoch=True)

        for l1l in range(self.n_future_steps):
            self.log("test/l1_loss_{:d}".format(l1l), loss["loss_range_timestep"][l1l], on_epoch=True)

        self.log(
            "test/mean_chamfer_distance", loss["mean_chamfer_distance"], on_epoch=True
        )
        self.log(
            "test/final_chamfer_distance", loss["final_chamfer_distance"], on_epoch=True
        )

        self.chamfer_distances_tensor = torch.cat(
            (self.chamfer_distances_tensor, loss["chamfer_distances_tensor"]), dim=1
        )

        return loss

    def on_test_epoch_end(self):
        # Remove first row since it was initialized with zero
        self.chamfer_distances_tensor = self.chamfer_distances_tensor[:, 1:]
        n_steps, _ = self.chamfer_distances_tensor.shape
        mean = torch.mean(self.chamfer_distances_tensor, dim=1)
        std = torch.std(self.chamfer_distances_tensor, dim=1)
        q = torch.tensor([0.25, 0.5, 0.75])
        quantile = torch.quantile(self.chamfer_distances_tensor, q, dim=1)

        chamfer_distances = []
        for s in range(n_steps):
            chamfer_distances.append(self.chamfer_distances_tensor[s, :].tolist())
        print("Final size of CD: ", self.chamfer_distances_tensor.shape)
        print("Mean :", mean)
        print("Std :", std)
        print("Quantile :", quantile)

        testdir = os.path.join(self.cfg["LOG_DIR"], "test")
        if not os.path.exists(testdir):
            os.makedirs(testdir)

        filename = os.path.join(
            testdir, "stats_" + time.strftime("%Y%m%d_%H%M%S") + ".yml"
        )

        log_to_save = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "quantile": quantile.tolist(),
            "chamfer_distances": chamfer_distances,
        }
        with open(filename, "w") as yaml_file:
            yaml.dump(log_to_save, yaml_file, default_flow_style=False)
