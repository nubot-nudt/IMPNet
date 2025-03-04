
import hdbscan
import hdbscan
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import copy
import open3d as o3d
from colorsys import hls_to_rgb
from impnet.utils.projection import projection
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import time

def gen_color_map(n):
  """ generate color map given number of instances
  """
  colors = []
  for i in np.arange(0., 360., 360. / n):
    h = i / 360.
    l = (50 + np.random.rand() * 10) / 100.
    s = (90 + np.random.rand() * 10) / 100.
    colors.append(hls_to_rgb(h, l, s))

  return np.array(colors)

class cluster_instance:
    def __init__(self,cfg):
      self.cfg = cfg
      self.projection = projection(self.cfg)
      self.clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=40, min_samples=None
                            )
    def get_clustered_point_id(self,output,b):

      masked_output = self.projection.get_moving_points_from_range_view(output)
      batch_size,n_future_steps,H,W = masked_output.shape
      instance_seg = []

      for s in range(n_future_steps): 
        mos_points = self.projection.get_valid_points_from_range_view(
                          masked_output[b,s,:,:]
                      ).view(-1, 3)

        device = mos_points.device
        if mos_points.shape[0] < 40:
          instance_seg.append([])
        else:
          instance_id_points = torch.from_numpy(\
                 self.clusterer.fit_predict(mos_points.cpu().detach().numpy())).to(device)
          instance_id_points = torch.cat((mos_points,instance_id_points.unsqueeze(1)),dim=1)
          mask = instance_id_points[:,-1]>=0
          instance_id_points = instance_id_points[mask>0]
          if (instance_id_points.shape[0]>0):
            instance_seg.append(instance_id_points.detach())
          else:
             instance_seg.append([])

      consistent_instance_seg = []

      consistent_instance_seg = make_instance_id_temporally_consistent(instance_seg,consistent_instance_seg)

      return consistent_instance_seg
    

    def get_instance_label(self,target,mos_label,b):

      n_future_steps = target.shape[1]
      instance_label_points = []
      for s in range(n_future_steps):         
        mos_gt_rv = self.projection.get_gt_moving_points_from_range_view(mos_label[b,s],target[b, s, 0, :, :])
        target_mos_points = target[b, s, 1:4, :, :].permute(1, 2, 0)
        target_mos_points = target_mos_points[mos_gt_rv  > 0.0].view(
                    1, -1, 3
                ) 
        if target_mos_points.shape[0] < 40:
          instance_label_points.append([])
        else :
          instance = self.clusterer.fit_predict(target_mos_points)
          instance = torch.cat((target_mos_points,instance.unsqueeze(1)),dim=1)
          instance_label_points.append(instance)
      return instance_label_points
    

def make_instance_id_temporally_consistent(pred_inst, consistent_instance_seg,matching_threshold=3.0):

    """
    输入每个batch的
    pred_inst(t,N,4)--x,y,z,id
    
    """


    T = len(pred_inst)
    # Initialise instance segmentations with prediction corresponding to the present
#0也是

    if len(pred_inst[0])>0:
      consistent_instance_seg.append(pred_inst[0])#只有id [t,（N，4）]
      largest_instance_id = consistent_instance_seg[-1][:,-1].max().item() 
      t_instance_ids = torch.unique(consistent_instance_seg[-1][:,-1])[0:].cpu().numpy() 
      center_last = found_instance_centers(pred_inst[0],t_instance_ids)
      for t in range(T-1):      
          if len(pred_inst[t+1]) == 0:
              # No instance so nothing to update
              consistent_instance_seg.append([])
              continue            
          t_instance_ids = torch.unique(pred_inst[t+1][:,-1])[0:].cpu().numpy() 
          if len(t_instance_ids) == 0:
              # No instance so nothing to update
              consistent_instance_seg.append([])
              continue
          
          center_now =  found_instance_centers(pred_inst[t + 1],t_instance_ids)
          
                     
          distances  = (cdist(center_now.cpu().numpy(),center_last.cpu().numpy(), metric='euclidean') )**0.5  

          ids_t, ids_t_one = linear_sum_assignment(distances) #new,old
          matching_distances = distances[ids_t, ids_t_one]
      


          #id_mapping = dict(zip(np.arange(0, len(t_instance_ids) ), t_instance_ids))
          #ids_t = np.vectorize(id_mapping.__getitem__, otypes=[np.int64])(ids_t)
          
          # Filter low quality match
          ids_t = ids_t[matching_distances < matching_threshold]
          ids_t_one = ids_t_one[matching_distances < matching_threshold]
          
          center_last = update_centers(center_last,center_now,old_ids=ids_t_one, new_ids=ids_t)#
          # Elements that are in t+1, but weren't matched
          remaining_ids = set(torch.unique(pred_inst[t + 1][:,-1]).cpu().numpy()).difference(set(ids_t))
          # remove background
          
          #  Set remaining_ids to a new unique id
          for remaining_id in list(remaining_ids):
              if remaining_id == -1:
                continue
              largest_instance_id += 1
              ids_t = np.append(ids_t, int(remaining_id))
              ids_t_one = np.append(ids_t_one, int(largest_instance_id))
              i = (np.where(t_instance_ids==remaining_id))[0]
              center_last =torch.cat((center_last,center_now[i,:]),dim=0)

          instance = update_instance_ids(pred_inst[t + 1], old_ids=ids_t_one, new_ids=ids_t)
          if (instance.shape[0]>0):
            consistent_instance_seg.append(instance)
    else:
        if(len(pred_inst[1:])>0):
          consistent_instance_seg.append([])
          consistent_instance_seg = (make_instance_id_temporally_consistent(pred_inst[1:],consistent_instance_seg))
        else:
           consistent_instance_seg.append([])
    return consistent_instance_seg 

def found_instance_centers(pred_inst_t,t_instance_ids):
    centers=[]
    for instance_id in t_instance_ids: #计算上一时刻的中心
        center = torch.zeros(1,3)
        instance_mask = (pred_inst_t[:,-1] == instance_id)
            
        points = pred_inst_t[instance_mask, :3]  # 取x,y,z坐标
        center[:,0] = points[:, 0].mean()
        center[:,1] = points[:, 1].mean()
        center[:,2] = points[:, 2].mean()
        centers.append(center)
    centers = torch.cat(centers,dim=0) #(t_instance_ids,3)    
    return centers        
def update_instance_ids(instance_seg, old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """
    indices = torch.arange(new_ids.max() + 1,device=instance_seg.device)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[new_id] = old_id
    mask = instance_seg[:,-1] >= 0 
    longi = instance_seg
    longi=longi.long()
    longi[mask,-1] = indices[longi[mask,-1].long()]
    instance_seg [mask,-1] =  longi[mask,-1].float()          
    return instance_seg

def update_centers(center_last,center_now ,old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """

    for old_id, new_id in zip(old_ids, new_ids):
        center_last[old_id,:] = center_now[new_id,:]
       
    return center_last


def gen_color_map(n):
  """ generate color map given number of instances
  """
  colors = []
  for i in np.arange(0., 360., 360. / n):
    h = i / 360.
    l = (50 + np.random.rand() * 10) / 100.
    s = (90 + np.random.rand() * 10) / 100.
    colors.append(hls_to_rgb(h, l, s))

  return np.array(colors)


def found_instance_bounding_box(pred_inst_t, t_instance_ids, boundingboxes):
    i = 0
    for instance_id in t_instance_ids:
        # 获取当前实例的点云
        instance_mask = (pred_inst_t[:, -1] == instance_id)
        points = pred_inst_t[instance_mask, :3]  # 只取前3列 (x, y, z)
        data = points.cpu().numpy()  # 转换为 numpy 数组

        # 直接计算点云的最大最小点
        xmin, ymin, zmin = np.min(data, axis=0)
        xmax, ymax, zmax = np.max(data, axis=0)

        # 计算边界框的中心点
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2

        # 计算边界框的尺寸
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin

        # 填充边界框信息
        boundingbox = np.zeros(7)
        boundingbox[0] = center_x  # 中心点 x
        boundingbox[1] = center_y  # 中心点 y
        boundingbox[2] = center_z  # 中心点 z
        boundingbox[3] = size_x    # 尺寸 x
        boundingbox[4] = size_y    # 尺寸 y
        boundingbox[5] = size_z    # 尺寸 z
        boundingbox[6] = 0         # 旋转角度（默认为 0，因为没有旋转）

        # 将结果保存到 boundingboxes 中
        boundingboxes[i, :] = torch.from_numpy(boundingbox).to(boundingboxes.device)
        i += 1

    return boundingboxes   