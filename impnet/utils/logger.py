#!/usr/bin/env python3
# @brief:    Logging and saving point clouds and range images
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import os
import torch
import matplotlib
import cmocean
import open3d as o3d
import numpy as np
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


point_size_config = {"material": {"cls": "PointsMaterial", "size": 0.1}}


def log_point_clouds(
    logger, projection, current_epoch, batch, output, sample_index, sequence, frame
):
    """Log point clouds to tensorboard

    Args:
        logger (TensorBoardLogger): Logger instance
        projection (projection): Projection instance
        current_epoch (int): Current epoch
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        sample_index ([int): Selected sample in batch
        sequence (int): Selected dataset sequence
        frame (int): Selected frame number
    """
    for step in [0, 4]:
        gt_points, pred_points, gt_colors, pred_colors = get_pred_and_gt_point_cloud(
            projection, batch, output, step, sample_index
        )
        concat_points = torch.cat(
            (gt_points.view(1, -1, 3), pred_points.view(1, -1, 3)), 1
        )
        concat_colors = torch.cat((gt_colors, pred_colors), 1)
        logger.add_mesh(
            "prediction_sequence_"
            + str(sequence)
            + "_frame_"
            + str(frame)
            + "_step_"
            + str(step),
            vertices=concat_points,
            colors=concat_colors,
            global_step=current_epoch,
            config_dict=point_size_config,
        )

import matplotlib.pyplot as plt
def save_point_cloud_with_instances(cfg,projection,range_views, instancesegs,sequence,frame):
    
    
    
    
    instance_ids =np.concatenate([tensor[:, -1].cpu().numpy() for tensor in instancesegs])
    unique_ids = np.unique(instance_ids)
    color_map =  create_color_map(unique_ids)
    for i in range(range_views.shape[0])   :
        path = make_path(
                os.path.join(
                    cfg["LOG_DIR"], "point_cloud_with_instances", str(sequence), str(frame),str(i)
                )
            )    
                
        range_view = range_views[i]
        instanceseg = instancesegs[i]
        print("111111111")
        save(projection,range_view, instanceseg,color_map,path)

    
def save(projection,range_view, instanceseg,color_map, output_path):

    # 创建3D图形的figure
    
    fig, ax = plt.subplots()
    points = (projection.get_valid_points_from_range_view(range_view)).cpu().numpy()
    # 绘制整体点云，使用较淡的颜色
    ax.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.5,s=0.05, label='Overall Point Cloud')
    instance_ids = (instanceseg[:,-1]).cpu().numpy()
    # 获取所有唯一的实例ID
    unique_ids = np.unique(instance_ids)

    # 遍历每个实例ID
    for id in unique_ids:
        if id == 0:  # 假设实例ID为0表示背景或未分类的点
            continue
        # 选择属于当前实例的点
        points_id = instanceseg[instance_ids == id].cpu().numpy()
        print("222222222222222")
        # 为每个实例生成一个随机颜色
        color = color_map[id]
        
        # 绘制当前实例的点
        ax.scatter(points_id[:, 0], points_id[:, 1], c=[color],s=0.05, label=f'Instance {id}')

    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    print("33333333333333")
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_color_map(unique_ids):
    """
    """
    color_map = {}
    for idx, uid in enumerate(unique_ids):
        # 使用固定算法根据索引生成颜色，例如使用hue循环
        hue = (idx % len(unique_ids)) / len(unique_ids)
        color = plt.cm.hsv(hue)  # 使用HSV颜色空间，这样颜色更加均匀
        color_map[uid] = color[:3]  # 取RGB值
    return color_map
def save_point_clouds(cfg, projection, batch, output):
    """Save ground truth and predicted point clouds as .ply

    Args:
        cfg (dict): Parameters
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
    """
    batch_size, n_future_steps, H, W = output["rv"].shape
    seq, frame = batch["meta"]
    n_past_steps = cfg["MODEL"]["N_PAST_STEPS"]

    for sample_index in range(batch_size):
        current_global_frame = frame[sample_index].item()

        path_to_point_clouds = os.path.join(
            '/root/autodl-tmp/test1',
            "test",
            "point_clouds",
            str(seq[sample_index].item()).zfill(2),
        )
        gt_path = make_path(os.path.join(path_to_point_clouds, "gt"))
        gt_move_path = make_path(os.path.join(path_to_point_clouds, "gt_mos"))
        pred_path = make_path(
            os.path.join(
                path_to_point_clouds, "pred", str(current_global_frame).zfill(6)
            )
        )
        pred_move_path = make_path(
            os.path.join(
                path_to_point_clouds, "pred_mos", str(current_global_frame).zfill(6)
            )
        )

        for step in range(n_future_steps):
            predicted_global_frame = current_global_frame + step + 1
            (
                gt_points,
                pred_points,
                pred_moving_points,
                gt_moving_points,
            ) = get_pred_and_gt_point_cloud(
                projection, batch, output, step, sample_index
            )
            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(
                gt_points.view(-1, 3).detach().cpu().numpy()
            )

            # 将运动点云添加到 moving_pcd
            moving_pcd = o3d.geometry.PointCloud()
            moving_pcd.points = o3d.utility.Vector3dVector(
                gt_moving_points.view(-1, 3).detach().cpu().numpy()
                )

            o3d.io.write_point_cloud(
                gt_path + "/" + str(predicted_global_frame).zfill(6) + ".ply", gt_pcd
            )
            o3d.io.write_point_cloud(
                gt_move_path + "/" + str(predicted_global_frame).zfill(6) + ".ply", moving_pcd
            )

            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(
                pred_points.view(-1, 3).detach().cpu().numpy()
            )
            pred_moving_pcd = o3d.geometry.PointCloud()
            pred_moving_pcd.points = o3d.utility.Vector3dVector(
                pred_moving_points.view(-1, 3).detach().cpu().numpy()
                )

            o3d.io.write_point_cloud(
                pred_path + "/" + str(predicted_global_frame).zfill(6) + ".ply",
                pred_pcd,
            )
            o3d.io.write_point_cloud(
                pred_move_path + "/" + str(predicted_global_frame).zfill(6) + ".ply",
                pred_moving_pcd,
            )


def get_pred_and_gt_point_cloud(projection, batch, output, step, sample_index):
    """Extract GT and predictions from batch and output dicts

    Args:
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        step (int): Prediction step
        sample_index ([int): Selected sample in batch

    Returns:
        list: GT and predicted points and colors
    """
    future_range = batch["fut_data"][sample_index, step, 0, :, :]
    mos_label = batch["mos_label"][sample_index, step, :, :]
    masked_prediction = projection.get_masked_range_view(output)
    
    gt_points = batch["fut_data"][sample_index, step,1:4, :, :].permute(1, 2, 0)
    gt_points = gt_points[future_range > 0.0].view(1, -1, 3)
    gt_colors = torch.zeros(gt_points.view(1, -1, 3).shape)
    gt_colors[:, :, 2] = 255

    pred_points = projection.get_valid_points_from_range_view(
        masked_prediction[sample_index, step, :, :]
    ).view(1, -1, 3)
    pred_colors = torch.zeros(pred_points.view(1, -1, 3).shape)
    pred_colors[:, :, 2] = 255

    pred_moving_points_mask = projection.get_moving_points_from_range_view(output)
    pred_moving_points = projection.get_valid_points_from_range_view(
        pred_moving_points_mask[sample_index, step, :, :]
    ).view(1, -1, 3)
    pred_moving_colors = torch.zeros(pred_moving_points.view(1, -1, 3).shape)
    pred_moving_colors[:, :, 0] = 255
    
    gt_points = batch["fut_data"][sample_index, step,1:4, :, :].permute(1, 2, 0)
    gt_moving_points = gt_points[mos_label>0].view(1, -1, 3)
    gt_moving_colors = torch.zeros(gt_moving_points.view(1, -1, 3).shape)
    gt_moving_colors[:, :, 0] = 255
    return gt_points, pred_points, pred_moving_points,gt_moving_points


def save_range_and_mask(cfg, projection, batch, past,future,output, sample_index, sequence, frame):
    """Saves GT and predicted range images and masks to a file

    Args:

        cfg (dict): Parameters
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        sample_index ([int): Selected sample in batch
        sequence (int): Selected dataset sequence
        frame (int): Selected frame number
    """
    cmap = mpl.colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=1, vmax=85, clip=False)
    cmap.set_under('k')
    _, n_past_steps, _, H, W = batch["past_data"].shape
    _, n_future_steps, _, _, _ = batch["fut_data"].shape


    min_range = -1.0  # due to invalid points
    max_range = cfg["DATA_CONFIG"]["MAX_RANGE"]

    past_range = past[sample_index, :, 0, :, :].view(n_past_steps, H, W)
    future_range = future[sample_index, :, 0, :, :].view(
        n_future_steps, H, W
    )
    #future_object_mask = batch["fut_data"][sample_index, 4, :, :, :].view(
    #    n_future_steps, H, W
    #)
    #future_ground_mask = torch.logical_not(future_object_mask)

    pred_rv = output["rv"][sample_index, :, :, :].view(n_future_steps, H, W)

    # Get masks
    past_mask = projection.get_target_mask_from_range_view(past_range)
    future_mask = projection.get_target_mask_from_range_view(future_range)
    pred_mask = projection.get_mask_from_output(output)[sample_index, :, :, :].view(
        n_future_steps, H, W
    )

    concat_pred_mask = torch.cat(
        (torch.zeros(past_mask.shape).type_as(past_mask), pred_mask), 0
    )
    concat_gt_mask = torch.cat((past_mask, future_mask), 0)

    # Get normalized range views
    concat_gt_rv = torch.cat((past_range, future_range), 0)
    concat_gt_rv_normalized = (concat_gt_rv - min_range) / (max_range - min_range)

    concat_pred_rv = torch.cat(
        (torch.zeros(past_range.shape).type_as(past_range), pred_rv), 0
    )
    concat_pred_rv_normalized = (concat_pred_rv - min_range) / (max_range - min_range)

    # Combine mask and rv predition
    masked_prediction = projection.get_masked_range_view(output)[
        sample_index, :, :, :
    ].view(n_future_steps, H, W)
    concat_combined_pred_rv = torch.cat(
        (torch.zeros(past_range.shape).type_as(past_range), masked_prediction), 0
    )
    concat_combined_pred_rv_normalized = (concat_combined_pred_rv - min_range) / (
        max_range - min_range
    )

    for s in range(0, n_past_steps + n_future_steps):
        step = "{0:02d}".format(s)

        # Save rv and mask predictions
        path = make_path(
            os.path.join(
                cfg["LOG_DIR"], "range_view_predictions1", str(sequence), str(frame)
            )
        )
        ratio = 5 * H / W
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(30, 30 * ratio))
        axs[0].imshow(concat_gt_rv[s, :, :].cpu().detach().numpy(), cmap=cmap, norm=norm)
        axs[0].text(
            0.01,
            0.8,
            "GT",
            transform=axs[0].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        axs[1].imshow(
            concat_combined_pred_rv[s, :, :].cpu().detach().numpy(), cmap=cmap, norm=norm
        )
        axs[1].text(
            0.01,
            0.8,
            "Pred",
            transform=axs[1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        axs[2].imshow(
            np.abs(
                concat_combined_pred_rv[s, :, :].cpu().detach().numpy()
                - concat_gt_rv[s, :, :].cpu().detach().numpy()), cmap=cmap, norm=norm
        )
        axs[2].text(
            0.01,
            0.8,
            "Range Loss",
            transform=axs[2].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        axs[0].set_title(
            "Step "
            + step
            + " of sequence "
            + str(sequence)
            + " from frame "
            + str(frame)
        )
        plt.savefig(path + "/" + step + ".png", bbox_inches="tight", transparent=True) #, dpi=2000)
        plt.close(fig)

def save_instace(cfg, projection, batch, output):
    """Save ground truth and predicted point clouds as .ply

    Args:
        cfg (dict): Parameters
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
    """
    batch_size, n_future_steps, H, W = output["rv"].shape
    seq, frame = batch["meta"]
    n_past_steps = cfg["MODEL"]["N_PAST_STEPS"]

    for sample_index in range(batch_size):
        current_global_frame = frame[sample_index].item()

        path_to_point_clouds = os.path.join(
            '/root/autodl-tmp/test1',
            "test",
            "point_clouds",
            str(seq[sample_index].item()).zfill(2),
        )
        gt_path = make_path(os.path.join(path_to_point_clouds, "gt"))
        gt_move_path = make_path(os.path.join(path_to_point_clouds, "gt_mos"))
        pred_path = make_path(
            os.path.join(
                path_to_point_clouds, "pred", str(current_global_frame).zfill(6)
            )
        )
        pred_move_path = make_path(
            os.path.join(
                path_to_point_clouds, "pred_mos", str(current_global_frame).zfill(6)
            )
        )

        for step in range(n_future_steps):
            predicted_global_frame = current_global_frame + step + 1
            (
                gt_points,
                pred_points,
                pred_moving_points,
                gt_moving_points,
            ) = get_pred_and_gt_point_cloud(
                projection, batch, output, step, sample_index
            )
            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(
                gt_points.view(-1, 3).detach().cpu().numpy()
            )

            # 将运动点云添加到 moving_pcd
            moving_pcd = o3d.geometry.PointCloud()
            moving_pcd.points = o3d.utility.Vector3dVector(
                gt_moving_points.view(-1, 3).detach().cpu().numpy()
                )

            o3d.io.write_point_cloud(
                gt_path + "/" + str(predicted_global_frame).zfill(6) + ".ply", gt_pcd
            )
            o3d.io.write_point_cloud(
                gt_move_path + "/" + str(predicted_global_frame).zfill(6) + ".ply", moving_pcd
            )

            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(
                pred_points.view(-1, 3).detach().cpu().numpy()
            )
            pred_moving_pcd = o3d.geometry.PointCloud()
            pred_moving_pcd.points = o3d.utility.Vector3dVector(
                pred_moving_points.view(-1, 3).detach().cpu().numpy()
                )

            o3d.io.write_point_cloud(
                pred_path + "/" + str(predicted_global_frame).zfill(6) + ".ply",
                pred_pcd,
            )
            o3d.io.write_point_cloud(
                pred_move_path + "/" + str(predicted_global_frame).zfill(6) + ".ply",
                pred_moving_pcd,
            )