from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import torch
import environment.transformations as tra
# Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet


def compute_stats(pred_transforms, data_loader):
    metrics_for_iter = defaultdict(list)
    num_processed = 0
    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, pred_transforms.device)

        batch_size = data['points_src'].shape[0]
        cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size]
        metrics = compute_metrics(data, cur_pred_transforms)
        for k in metrics:
            metrics_for_iter[k].append(metrics[k])
        num_processed += batch_size
    summary_metrics = summarize_metrics(metrics_for_iter)

    return metrics_for_iter, summary_metrics


def compute_metrics(data, pred_transforms):
    gt_transforms = data['transform_gt']
    igt_transforms = torch.eye(4, device=pred_transforms.device).repeat(gt_transforms.shape[0], 1, 1)
    igt_transforms[:, :3, :3] = gt_transforms[:, :3, :3].transpose(2, 1)
    igt_transforms[:, :3, 3] = -(igt_transforms[:, :3, :3] @ gt_transforms[:, :3, 3].view(-1, 3, 1)).view(-1, 3)
    points_src = data['points_src'][..., :3]
    points_ref = data['points_ref'][..., :3]
    if 'points_raw' in data:
        points_raw = data['points_raw'][..., :3]
    else:
        points_raw = points_ref

    # Euler angles, Individual translation errors (Deep Closest Point convention)
    r_gt_euler_deg = np.stack([Rotation.from_dcm(r.cpu().numpy()).as_euler('xyz', degrees=True)
                               for r in gt_transforms[:, :3, :3]])
    r_pred_euler_deg = np.stack([Rotation.from_dcm(r.cpu().numpy()).as_euler('xyz', degrees=True)
                                 for r in pred_transforms[:, :3, :3]])
    t_gt = gt_transforms[:, :3, 3]
    t_pred = pred_transforms[:, :3, 3]
    r_mae = np.abs(r_gt_euler_deg - r_pred_euler_deg).mean(axis=1)
    t_mae = torch.abs(t_gt - t_pred).mean(dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = igt_transforms @ pred_transforms
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    r_iso = torch.rad2deg(torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)))
    t_iso = concatenated[:, :3, 3].norm(dim=-1)

    # Modified Chamfer distance
    src_transformed = (pred_transforms[:, :3, :3] @ points_src.transpose(2, 1)).transpose(2, 1)\
                      + pred_transforms[:, :3, 3][:, None, :]
    ref_clean = points_raw
    residual_transforms = pred_transforms @ igt_transforms
    src_clean = (residual_transforms[:, :3, :3] @ points_raw.transpose(2, 1)).transpose(2, 1)\
                + residual_transforms[:, :3, 3][:, None, :]
    dist_src = torch.min(tra.square_distance(src_transformed, ref_clean), dim=-1)[0]
    dist_ref = torch.min(tra.square_distance(points_ref, src_clean), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # ADD/ADI
    src_diameters = torch.sqrt(tra.square_distance(src_clean, src_clean).max(dim=-1)[0]).max(dim=-1)[0]
    dist_add = torch.norm(src_clean - ref_clean, p=2, dim=-1).mean(dim=1) / src_diameters
    dist_adi = torch.sqrt(tra.square_distance(ref_clean, src_clean)).min(dim=-1)[0].mean(dim=-1) / src_diameters

    metrics = {
        'r_mae': r_mae,
        't_mae': t_mae.cpu().numpy(),
        'r_iso': r_iso.cpu().numpy(),
        't_iso': t_iso.cpu().numpy(),
        'chamfer_dist': chamfer_dist.cpu().numpy(),
        'add': dist_add.cpu().numpy(),
        'adi': dist_adi.cpu().numpy()
    }
    return metrics


def summarize_metrics(metrics):
    summarized = {}
    for k in metrics:
        metrics[k] = np.hstack(metrics[k])
        if k.startswith('ad'):
            step_precision = 1e-3
            max_precision = 0.1
            precisions = np.arange(step_precision, max_precision + step_precision, step_precision)
            recalls = np.array([(metrics[k] <= precision).mean() for precision in precisions])
            # integrate area under precision-recall curve -- normalize to 100% (= area given by 1.0 * max_precision)
            summarized[k + '_auc10'] = (recalls * step_precision).sum()/max_precision
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device
    via RPMNet """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)
            if tensor_dict[k].dtype == torch.double:
                tensor_dict[k] = tensor_dict[k].float()
        if isinstance(tensor_dict[k], dict):
            dict_all_to_device(tensor_dict[k], device)
