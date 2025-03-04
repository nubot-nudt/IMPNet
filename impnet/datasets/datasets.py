#!/usr/bin/env python3
# @brief:    Pytorch Lightning module for KITTI Odometry
# @author:   Kaustab Pal [kaustab21@gmail.com]
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import copy

from impnet.utils.projection import projection
from impnet.datasets.preprocess_data import prepare_data, compute_mean_and_std
from impnet.utils.utils import load_files

class KittiOdometryModule(pl.LightningDataModule):
    """A Pytorch Lightning module for KITTI Odometry"""

    def __init__(self, cfg):
        """Method to initizalize the Kitti Odometry dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(KittiOdometryModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        """Call prepare_data method to generate npy range images from raw LiDAR data"""
    
      #  prepare_data(self.cfg)

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        ########## Point dataset splits
        train_set = KittiOdometryRaw(self.cfg, split="train")

        val_set = KittiOdometryRaw(self.cfg, split="val")

        test_set = KittiOdometryRaw(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=self.cfg["DATA_CONFIG"]["DATALOADER"]["SHUFFLE"],
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers = self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True
        )
        self.test_iter = iter(self.test_loader)

        # Optionally compute statistics of training data
        if self.cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"]:
            compute_mean_and_std(self.cfg, self.train_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class KittiOdometryRaw(Dataset):
    """Dataset class for range image-based point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = self.cfg["PROCESSED_DATA"]
        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.n_channels = 4

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]

        self.num_class=self.cfg["MODEL"]["NUM_CLASS"]

        self.semantic_mos_config = yaml.safe_load(open(cfg["DATA_CONFIG"]["SEMANTIC_MOS_CONFIG_FILE"]))
        # Projection class for mapping from range image to 3D point cloud
        self.projection = projection(self.cfg)

        if split == "train":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        elif split == "val":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        elif split == "test":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")

        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames_range = {}
        self.filenames_xyz = {}
        self.filenames_intensity = {}
        self.filenames_semantic = {}
        self.filenames_instance={}
        self.filenames_gt_offset = {}
        self.filenames_gt_heatmap ={}
        for residual_idx in range(self.n_past_steps-1):
            exec("self.residual_files_" + str(residual_idx+1) + " = {}")
        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            scan_path_range = os.path.join(self.root_dir, seqstr, "processed", "range")
            self.filenames_range[seq] = load_files(scan_path_range)

            scan_path_xyz = os.path.join(self.root_dir, seqstr, "processed", "xyz")
            self.filenames_xyz[seq] = load_files(scan_path_xyz)

            scan_path_label = os.path.join(self.root_dir, seqstr, "processed", "labels")
            self.filenames_semantic[seq] =load_files(scan_path_label)
            
            scan_instance_label = os.path.join(self.root_dir, seqstr,"processed", "boundingbox")
            self.filenames_instance[seq] =load_files(scan_instance_label)

            for residual_idx in range(self.n_past_steps-1):
                folder_name = "residual_images_" + str(residual_idx+1)
                exec("residual_path_" + str(residual_idx+1) + "=" + "os.path.join(self.root_dir, seqstr, folder_name)")
                exec("residual_files_" + str(residual_idx+1) + " = " + '[os.path.join(dp, f) for dp, dn, fn in '
                        'os.walk(os.path.expanduser(residual_path_' + str(residual_idx+1) + '))'
                        ' for f in fn]')
                exec("residual_files_" + str(residual_idx+1) + ".sort()")
                exec("self.residual_files_" + str(residual_idx+1) + "[seq]" + " = " + "residual_files_" + str(residual_idx+1))
            
            # Get number of sequences based on number of past and future steps
            n_samples_sequence = max(
                0,
                len(self.filenames_range[seq])
                - self.n_past_steps
                - self.n_future_steps
                + 1,
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.n_past_steps + sample_idx - 1
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load and concatenate range image channels

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        seq, scan_idx = self.idx_mapper[idx]

        # Load past data
        past_data = torch.empty(
            [self.n_past_steps, self.n_channels, self.height, self.width]
        )
        res_data=torch.empty(
            [self.n_past_steps, self.height, self.width]
        )

        from_idx = scan_idx - self.n_past_steps + 1
        to_idx = scan_idx
        past_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        past_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        

        for t in range(self.n_past_steps):
            past_data[t, 0, :, :] = self.load_range(past_filenames_range[t])
            past_data[t, 1:4, :, :] = self.load_xyz(past_filenames_xyz[t])

        for t in range(self.n_past_steps-1):
            exec("residual_file_" + str(t+1) + " = " + "self.residual_files_" + str(t+1) + "[seq][scan_idx]")           
            exec("proj_residuals_" + str(t+1) + " = torch.Tensor(np.load(residual_file_" + str(t+1) + "))")
            exec("res_data["+str(self.n_past_steps-1-(t+1))+",:,:] = proj_residuals_" + str(t+1)) 
        current_numpy = np.zeros((64,2048))
        current= torch.tensor(current_numpy)
        res_data[4,:,:] =current       
        
        # Load future data
        fut_data = torch.empty(
            [self.n_future_steps, self.n_channels, self.height, self.width]
        )

        mos_label = torch.empty(
            [self.n_future_steps, self.height, self.width]
        )
        gt_boxes = torch.empty(
            [self.n_future_steps,500, 8]
        )

        
        from_idx = scan_idx + 1
        to_idx = scan_idx + self.n_future_steps
        fut_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        fut_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        fut_filenames_semantic = self.filenames_semantic[seq][from_idx : to_idx + 1] 
        fut_filenames_instance=self.filenames_instance[seq][from_idx : to_idx + 1] 


        gt_boxes = []
        for t in range(self.n_future_steps):
            fut_data[t, 0, :, :] = self.load_range(fut_filenames_range[t])
            fut_data[t, 1:4, :, :] = self.load_xyz(fut_filenames_xyz[t])           
            mos_label[t, :, :] = self.load_moslabel(fut_filenames_semantic[t])       
            list_bounding_box = [self.read_bounding_box_label(fut_filenames_instance[t])] 
            
            gt_box = np.zeros([500,8])
            for i, boxs in enumerate(list_bounding_box):
                gt_box[:len(boxs),0:7] = boxs[:,2:9]
                gt_box[:len(boxs),7] = boxs[:,0]
                gt_box = torch.from_numpy(gt_box)
            gt_boxes.append(gt_box)
        gt_boxes =torch.stack(gt_boxes,dim=0)

        

        item = {"past_data": past_data, "mos_label": mos_label,"fut_data": fut_data, "res_data":res_data,
                "meta": (seq, scan_idx),"instance_label":gt_boxes}
        return item
    
    def read_bounding_box_label(self,filename):
        """Load object boundingbox  from .npy file"""
        boundingbox_label_load = np.load(filename,allow_pickle=True)
        
        if len(boundingbox_label_load)==0: 
            boundingbox_label_load = []
            boundingbox_label_load.append([0,0,1,[0,0,0,0,0,0,0]]) 
        dynamic_falg = False
        boundingbox_label_list = []
        for i in range(0,len(boundingbox_label_load)):
            boundingbox_label = np.zeros(9,dtype=np.float32)
            boundingbox_label[0] = boundingbox_label_load[i][1]
            boundingbox_label[1] = boundingbox_label_load[i][2]
            boundingbox_label[2:9] = boundingbox_label_load[i][3][:]

            boundingbox_label_list.append(boundingbox_label)



        box_label_numpy = np.array(boundingbox_label_list)
        
        return box_label_numpy
    
    def load_moslabel(self, filename):
        slabel = np.load(filename)
        mapped_labels = copy.deepcopy(slabel)
        for k, v in self.semantic_mos_config["learning_map"].items():
            mapped_labels[slabel == k] = v
        mos_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
        return mos_labels  

    def load_inslabel(self, filename):

        ins_labels = torch.Tensor(np.load(filename)).long()
        return ins_labels  

    def load_range(self, filename):
        """Load .npy range image as (1,height,width) tensor"""
        rv = torch.Tensor(np.load(filename)).float()
        return rv


    def load_heatmap(self, filename):
        """Load .npy range image as (1,height,width) tensor"""
        rv = torch.Tensor(np.load(filename)).float()
        return rv
    def load_xyz(self, filename):
        """Load .npy xyz values as (3,height,width) tensor"""
        xyz = torch.Tensor(np.load(filename)).float()[:, :, :3]
        xyz = xyz.permute(2, 0, 1)
        return xyz

if __name__ == "__main__":
    config_filename = "./config/parameters.yml"
    cfg = yaml.safe_load(open(config_filename))
    data = KittiOdometryModule(cfg)
    data.prepare_data()
    data.setup()

    item = data.valid_loader.dataset.__getitem__(0)

    def normalize(image):
        min = np.min(image)
        max = np.max(image)
        normalized_image = (image - min) / (max - min)
        return normalized_image

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(30, 30 * 5 * 64 / 2048))

    axs[0].imshow(normalize(item["fut_data"][0, 0, :, :].numpy()))
    axs[0].set_title("Range")
    axs[1].imshow(normalize(item["fut_data"][1, 0, :, :].numpy()))
    axs[1].set_title("X")
    axs[2].imshow(normalize(item["fut_data"][2, 0, :, :].numpy()))
    axs[2].set_title("Y")
    axs[3].imshow(normalize(item["fut_data"][3, 0, :, :].numpy()))
    axs[3].set_title("Z")
    axs[4].imshow(normalize(item["fut_data"][4, 0, :, :].numpy()))
    axs[4].set_title("Intensity")

    plt.show()
