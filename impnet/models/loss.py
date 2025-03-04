import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random
import time
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
from impnet.utils.projection import projection
from impnet.utils.Lovasz_Softmax import Lovasz_softmax
from impnet.utils.utils import map
from impnet.models.cluster import found_instance_bounding_box
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
class Loss(nn.Module):
    """Combined loss for point cloud prediction"""

    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg
        self.use_instance = self.cfg["MODEL"]["INS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.loss_weight_cd = self.cfg["TRAIN"]["LOSS_WEIGHT_CHAMFER_DISTANCE"]
        self.loss_weight_rv = self.cfg["TRAIN"]["LOSS_WEIGHT_RANGE_VIEW"]
        self.loss_weight_mask = self.cfg["TRAIN"]["LOSS_WEIGHT_MASK"]
        self.loss_weight_mos = self.cfg["TRAIN"]["LOSS_WEIGHT_MOS"]

        self.loss_range = loss_range(self.cfg)
        self.chamfer_distance = chamfer_distance(self.cfg)
        self.loss_mask = loss_mask(self.cfg)
        self.loss_mos = loss_mos(self.cfg)
        self.loss_instance = loss_instance_box(self.cfg)

    def forward(self, output, target, proj_labels,intance_labels,mode, epoch_number=40):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """

        target_range_image = target[:,:, 0, :, :]

        # Range view
        loss_range_view, loss_range_timestep = self.loss_range(output, target_range_image)

        # Mask
        loss_mask = self.loss_mask(output, target_range_image)
        
        #Motion Sem
        #loss_mos = self.loss_mos(output,proj_labels)

        #los_instance
        loss_instance = torch.tensor(0.0, requires_grad=True,device = target_range_image.device)
        

        
        # Chamfer Distance
        if epoch_number>=50 or self.loss_weight_cd > 0.0 or mode == "val" or mode == "test":
            chamfer_distance, chamfer_distances_tensor = self.chamfer_distance(
                output, target,proj_labels , self.cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"]
            )
            loss_chamfer_distance = sum([cd for cd in chamfer_distance.values()]) / len(
                chamfer_distance
            )
            detached_chamfer_distance = {
                step: cd.detach() for step, cd in chamfer_distance.items()
            }

            
        else:
            chamfer_distance = dict(
                (step, torch.zeros(1).type_as(target_range_image))
                for step in range(self.n_future_steps)
            )
            chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
            loss_chamfer_distance = torch.zeros_like(loss_range_view)
            detached_chamfer_distance = chamfer_distance
        if self.use_instance:
            for b in range(target.shape[0]):
                loss_instance = loss_instance+self.loss_instance(output,intance_labels,b)



        loss = (
            self.loss_weight_cd * (loss_chamfer_distance)
            + self.loss_weight_rv * loss_range_view
            + self.loss_weight_mask * loss_mask
            + self.loss_weight_mos * loss_mos
            +loss_instance
        )
        
        loss_dict = {
            "loss": loss,
            "chamfer_distance": detached_chamfer_distance,
            "chamfer_distances_tensor": chamfer_distances_tensor.detach(),
            "mean_chamfer_distance": loss_chamfer_distance.detach(),
            "final_chamfer_distance": chamfer_distance[
                self.n_future_steps - 1
            ].detach(),
            "loss_range_view": loss_range_view.detach(),
            "loss_range_timestep": loss_range_timestep.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_mos" :loss_mos.detach(),
            "loss_instance":loss_instance
        }
        return loss_dict


class loss_mask(nn.Module):
    """Binary cross entropy loss for prediction of valid mask"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.projection = projection(self.cfg)

    def forward(self, output, target_range_view):
        target_mask = self.projection.get_target_mask_from_range_view(target_range_view)
        loss = self.loss(output["mask_logits"], target_mask)
        return loss

class loss_mos(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.DATA = yaml.safe_load(open(cfg["DATA_CONFIG"]["SEMANTIC_MOS_CONFIG_FILE"]))
        self.set_loss_weight()
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.criterion = nn.CrossEntropyLoss()#weight=self.loss_w.double())
        self.ls = Lovasz_softmax(ignore=None)

    def set_loss_weight(self):
        """
            Used to calculate the weights for each class
            weights for loss (and bias)
        """
        epsilon_w= 0.001       # class weight w = 1 / (content + epsilon_w)
        content = torch.zeros(2, dtype=torch.float) #num_class=2
        for cl, freq in self.DATA["content"].items():
            x_cl = map(cl,self.DATA["learning_map"])   # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):   # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:    # don't weigh
                self.loss_w[x_cl] = 0

    def forward(self,output,proj_labels):
        output = output["motion_seg"]
        b = output.shape[0]
        #output[target_range_image == -1.0] = -1.0
        output1 = output.permute(0,2,1,3,4)
        output2 = F.softmax(output,dim=2)       
        loss1 = self.criterion(output1.double(), proj_labels.long())
        loss2 = self.ls(output2.view(5*b,2,64,2048), (proj_labels.view(5*b,64,2048)).long())
        return loss1+loss2

class loss_instance_box(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.loss = nn.L1Loss(reduction="mean")
        
    def forward(self,output,instance_labels,b):

        #output[target_range_image == -1.0] = -1.0
        start = time.time()
        consistent_instance_seg = output['consistent_instance_seg'][b]
        instance_time = (time.time() - start) 
        #print("-------------------instance_time---------------------",instance_time)

        instance_label = instance_labels[b]   
        #print("-------------------label_time---------------------",label_time)
        timestep_loss = torch.zeros(instance_label.shape[0])
        for t in range(instance_label.shape[0]):
            instance_label_t = instance_label[t]
            instance_pred_t = consistent_instance_seg[t]

            if(len(instance_pred_t)>0):
                gt_num=  (torch.sum(instance_label_t[:, -1] > 0)).item()
                predict_num = torch.unique(instance_pred_t[:,-1]).shape[0]
                num = max(gt_num,predict_num)
            
                pred_box=torch.zeros(500,7)
                predict_ids = torch.unique(instance_pred_t[:,-1]).cpu().numpy() 
                pred_box = found_instance_bounding_box(consistent_instance_seg[t],predict_ids,pred_box)
                if num>0 :
                    pred_box = pred_box[:num,:]
                    gt_box = instance_label_t[:num,:7]   
                    distances  = (cdist(pred_box[:,:3].cpu().numpy(),gt_box[:,:3].cpu().numpy(), metric='euclidean') )
                    ids_t, ids_t_one = linear_sum_assignment(distances) 
                    sorted_gt_box = gt_box[ids_t_one]
                    sorted_pred_box = pred_box[ids_t].to(sorted_gt_box.device)         
                    #print('sorted_pred_box',sorted_pred_box)
                    #print('sorted_gt_box',sorted_gt_box)
                    
                    timestep_loss[t]=self.loss(sorted_gt_box,sorted_pred_box)
                    #print("loss",timestep_loss[t])
        loss = timestep_loss.sum()

        return loss

class loss_range(nn.Module):
    """L1 loss for range image prediction"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, output, target_range_image):
        # Do not count L1 loss for invalid GT points
        gt_masked_output = output["rv"].clone()
        gt_masked_output[target_range_image == -1.0] = -1.0
        loss = self.loss(gt_masked_output, target_range_image)
        timestep_loss = torch.zeros(target_range_image.shape[1])
        for i in range(target_range_image.shape[1]):
            timestep_loss[i] = self.loss(gt_masked_output[:, i, :, :], target_range_image[:, i, :, :])
        return loss, timestep_loss


class chamfer_distance(nn.Module):
    """Chamfer distance loss. Additionally, the implementation allows the evaluation
    on downsampled point cloud (this is only for comparison to other methods but not recommended,
    because it is a bad approximation of the real Chamfer distance.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target,mos_label, n_samples):
        batch_size, n_future_steps, H, W = output["rv"].shape
        masked_output = self.projection.get_masked_range_view(output)
        masked_mos_output = self.projection.get_moving_points_from_range_view(output)
        chamfer_distances = {}
        chamfer_distances_tensor = torch.zeros(n_future_steps, batch_size)
        for s in range(n_future_steps):
            chamfer_distances[s] = 0
            for b in range(batch_size):
                output_points = self.projection.get_valid_points_from_range_view(
                    masked_output[b, s, :, :]
                ).view(1, -1, 3)
                output_mos_points = self.projection.get_valid_points_from_range_view(
                    masked_mos_output[b, s, :, :]
                ).view(1, -1, 3)
                target_points = target[b, s, 1:4, :, :].permute(1, 2, 0)
                target_points = target_points[target[b, s, 0, :, :] > 0.0].view(
                    1, -1, 3
                )
                mos_gt_rv = self.projection.get_gt_moving_points_from_range_view(mos_label[b,s],target[b, s, 0, :, :])
                target_mos_points = target[b, s, 1:4, :, :].permute(1, 2, 0)
                target_mos_points = target_mos_points[mos_gt_rv  > 0.0].view(
                    1, -1, 3
                )                

                if n_samples != -1:
                    n_output_points = output_points.shape[1]
                    n_target_points = target_points.shape[1]
                    n_samples = min(n_samples, n_output_points, n_target_points)

                    sampled_output_indices = random.sample(
                        range(n_output_points), n_samples
                    )
                    sampled_target_indices = random.sample(
                        range(n_target_points), n_samples
                    )

                    output_points = output_points[:, sampled_output_indices, :]
                    target_points = target_points[:, sampled_target_indices, :]

                dist1, dist2 = self.loss(output_points, target_points)
                dist_combined = torch.mean(dist1) + torch.mean(dist2)
                n_mos_samples = min(target_mos_points.shape[1],output_mos_points.shape[1])                
                chamfer_distances[s] += dist_combined
                chamfer_distances_tensor[s, b] = dist_combined
            chamfer_distances[s] = chamfer_distances[s] / batch_size
        #print("chamfer_distances",chamfer_distances)
        return chamfer_distances, chamfer_distances_tensor

