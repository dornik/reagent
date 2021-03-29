import numpy as np
import time
import matplotlib.pyplot as plt
from drawnow import drawnow
import cv2 as cv
import torch
from tqdm import tqdm

import config as cfg

import os
import sys
sys.path.append(cfg.BOP_PATH)
sys.path.append(os.path.join(cfg.BOP_PATH, "bop_toolkit_lib"))
import bop_toolkit_lib.renderer as renderer


class CloudVisualizer:

    def __init__(self, t_sleep=0.5, point_size=3, ortho=True):
        self.t_sleep = t_sleep
        self.point_size = point_size
        self.ortho = ortho

        self.pcd_src_2d, self.pcd_tgt_2d, self.pcd_est = None, None, None

    def reset(self, pcd_src, pcd_tgt, pcd_est):
        self.pcd_src_2d = self.project(pcd_src[:, :3])
        self.pcd_tgt_2d = self.project(pcd_tgt[:, :3])
        self.pcd_est = pcd_est[:, :3]
        drawnow(self.plot)

    def update(self, new_est):
        self.pcd_est = new_est
        drawnow(self.plot)
        time.sleep(self.t_sleep)

    def plot(self):
        pcd_est_2d = self.project(self.pcd_est)

        # get scale and center
        yx_min, yx_max = np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).min(axis=0),\
                         np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).max(axis=0)
        dimensions = yx_max - yx_min
        center = yx_min + dimensions/2

        # get appropriate x/y axes limits
        dimensions = np.array([dimensions.max()*1.1]*2)
        yx_min = center - dimensions/2
        yx_max = center + dimensions/2

        cmap = plt.get_cmap('tab20')
        magenta, gray, cyan = cmap.colors[12], cmap.colors[14], cmap.colors[18]

        plt.scatter(self.pcd_src_2d[:, 0], self.pcd_src_2d[:, 1], c=np.asarray(magenta)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(self.pcd_tgt_2d[:, 0], self.pcd_tgt_2d[:, 1], c=np.asarray(gray)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(pcd_est_2d[:, 0], pcd_est_2d[:, 1], c=np.asarray(cyan)[None, :],
                    s=self.point_size, alpha=0.7)
        plt.xlim([yx_min[0], yx_max[0]])
        plt.ylim([yx_min[1], yx_max[1]])
        plt.xticks([])
        plt.yticks([])

    def project(self, points):
        Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2] + 3.0  # push back a little
        if self.ortho:
            xs = Xs
            ys = Ys
        else:
            xs = np.divide(Xs, Zs)
            ys = np.divide(Ys, Zs)
        points_2d = np.hstack([xs[:, None], ys[:, None]])
        return points_2d

    def capture(self, path):
        plt.savefig(path, dpi='figure')


class OutlineVisualizer:

    def __init__(self, dataset_name, obj_ids, t_sleep=0.5):
        self.t_sleep = t_sleep

        # -- set-up renderer for mask propagation
        self.ren = renderer.create_renderer(640, 480, renderer_type='python', mode='depth')
        print("  loading models into BOP renderer...")
        for obj_id in tqdm(obj_ids):
            obj_path = os.path.join(cfg.LM_PATH, f"models/obj_{obj_id:06d}.ply")
            self.ren.add_object(obj_id, obj_path)

    def reset(self, data, rgb, split='test', show_init=True):
        # --- prepare inputs
        self.split = split
        # ground truth
        self.obj_id = int(data['gt']['obj_id'][0])
        self.gt_m2c = torch.eye(4, device=data['est'].device).repeat(data['est'].shape[0], 1, 1)
        self.gt_m2c[:, :3, :3] = data['gt']['cam_R_m2c']
        self.gt_m2c[:, :3, 3] = data['gt']['cam_t_m2c'].squeeze()
        self.K = data['cam']['cam_K'][0].cpu().numpy()
        # normalization
        self.scale = data['normalization'][:, 0, 0]
        self.offset = data['normalization'][:, :3, 3]
        # RGB observation
        self.rgb = rgb
        self.observation = data['points_src'][..., :3].clone().cpu()

        # --- prepare point clouds
        if split == 'test':  # -> undo normalization
            self.observation[..., :3] = self.observation[..., :3] * self.scale[:, None, None] + self.offset[:, None, :]
            self.init_o2c = data['est']  # observation to camera space (ground truth)
            self.init_m2c = data['est']  # erroneous model to camera space (initial error)
        else:  # -> undo error in normalized coords, then undo normalization
            self.observation[..., :3] = (data['transform_gt'][:, :3, :3] @ self.observation[..., :3].transpose(2, 1))\
                                            .transpose(2, 1) + data['transform_gt'][:, :3, 3][:, None, :]
            self.observation[..., :3] = self.observation[..., :3] * self.scale[:, None, None] + self.offset[:, None, :]
            self.init_o2c = data['est']  # observation to camera space (ground truth)
            # erroneous model to camera space (initial error)
            error = torch.eye(4, device=data['est'].device).repeat(data['est'].shape[0], 1, 1)
            error[:, :3, :] = data['transform_gt']
            error[:, :3, 3] *= self.scale[:, None]
            self.init_m2c = data['est'] @ error

        # --- prepare outlines for plotting
        self.gt_contour, self.gt_center, self.gt_dimensions = self.get_contour(self.obj_id, self.gt_m2c)
        self.init_contour, _, _ = self.get_contour(self.obj_id, self.init_m2c)
        init_o2i = data['cam']['cam_K'].float() @ self.init_o2c[:, :3, :]  # to image space
        proj_src = (init_o2i[:, :3, :3] @ self.observation[..., :3].transpose(2, 1)).transpose(2, 1)\
                   + init_o2i[:, :3, 3][:, None, :]
        self.proj_src = torch.round(proj_src / proj_src[..., 2][..., None])[0, :, :2].cpu().numpy()

        if show_init:
            self.update(torch.eye(4, device=data['est'].device).repeat(data['est'].shape[0], 1, 1))

    def update(self, est_init2m):
        # pose
        est_init2m[:, :3, 3] *= self.scale[:, None]
        # equiv. to: est_m2c = metrics.invert_tensor(est_init2m @ metrics.invert_tensor(self.init_m2c))
        est_m2init = torch.eye(4, device=est_init2m.device).repeat(est_init2m.shape[0], 1, 1)
        est_m2init[:, :3, :3] = est_init2m[:, :3, :3].transpose(2, 1)
        est_m2init[:, :3, 3] = -(est_m2init[:, :3, :3] @ est_init2m[:, :3, 3].view(-1, 3, 1)).view(-1, 3)
        est_m2c = self.init_m2c @ est_m2init
        # plot
        drawnow(self.plot, est_m2c=est_m2c)
        time.sleep(self.t_sleep)

    def plot(self, est_m2c):
        est_contour, _, _ = self.get_contour(self.obj_id, est_m2c)

        magenta, gray, cyan = (1, 0, 1), (0.7, 0.7, 0.7), (0, 0.9, 1.0)

        vis = self.rgb.copy()
        vis = cv.drawContours(vis, self.gt_contour, -1, gray, 1, lineType=cv.LINE_AA)
        vis = cv.drawContours(vis, self.init_contour, -1, magenta, 1, lineType=cv.LINE_AA)
        vis = cv.drawContours(vis, est_contour, -1, cyan, 1, lineType=cv.LINE_AA)

        # crop
        focus = self.gt_center
        halfsize = self.gt_dimensions.max()/2 * 1.2

        # plot
        plt.subplot(1, 2, 1)
        plt.imshow(self.rgb)
        plt.scatter(self.proj_src[:, 0], self.proj_src[:, 1], c=np.asarray(magenta)[None, :], s=2, alpha=0.2)
        plt.plot([focus[1] - halfsize, focus[1] - halfsize, focus[1] + halfsize, focus[1] + halfsize,
                  focus[1] - halfsize],
                 [focus[0] + halfsize, focus[0] - halfsize, focus[0] - halfsize, focus[0] + halfsize,
                  focus[0] + halfsize],
                 'k-', linewidth=2)
        plt.xlim([80, 640-80])
        plt.title("Sampled points")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(vis)
        plt.xlim([focus[1] - halfsize, focus[1] + halfsize])
        plt.ylim([focus[0] + halfsize, focus[0] - halfsize])
        plt.title("Pose estimate (zoomed-in)")
        plt.axis('off')

    def get_contour(self, obj_id, m2c):
        fx, fy, cx, cy = float(self.K[0, 0]), float(self.K[1, 1]), float(self.K[0, 2]), float(self.K[1, 2])
        R, t = m2c[0, :3, :3].cpu().numpy(), m2c[0, :3, 3].cpu().numpy()
        depth = self.ren.render_object(obj_id, R, t, fx, fy, cx, cy)['depth']
        yx_min, yx_max = np.argwhere(depth > 0).min(axis=0), np.argwhere(depth > 0).max(axis=0)
        dimensions = yx_max - yx_min
        center = dimensions/2 + yx_min
        contour, _ = cv.findContours(np.uint8(depth > 0), cv.RETR_CCOMP,
                                     cv.CHAIN_APPROX_TC89_L1)
        return contour, center, dimensions

    def capture(self, path):
        plt.savefig(path, dpi='figure')
