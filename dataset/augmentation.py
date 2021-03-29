import math
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data
import transforms3d as t3d
# Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        if 'points' in sample:
            sample['points_raw'] = sample.pop('points')
        else:
            assert 'points_raw' in sample
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05, only_ref=False):
        self.scale = scale
        self.clip = clip
        self.only_ref = only_ref

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            if not self.only_ref:
                sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class TransformSE3:
    def __init__(self):
        """Applies a random rigid transformation to the source point cloud"""

    def apply_transform(self, p0, transform_mat):
        p1 = (transform_mat[:3, :3] @ p0[:, :3].T).T + transform_mat[:3, 3]
        if p0.shape[1] >= 6:  # Need to rotate normals too
            n1 = (transform_mat[:3, :3] @ p0[:, 3:6].T).T
            p1 = np.concatenate((p1, n1), axis=-1)
        if p0.shape[1] == 4:  # label (pose estimation task)
            p1 = np.concatenate((p1, p0[:, 3][:, None]), axis=-1)
        if p0.shape[1] > 6:  # additional channels after normals
            p1 = np.concatenate((p1, p0[:, 6:]), axis=-1)

        igt = transform_mat
        # invert to get gt
        gt = igt.copy()
        gt[:3, :3] = gt[:3, :3].T
        gt[:3, 3] = -gt[:3, :3] @ gt[:3, 3]

        return p1, gt, igt

    def __call__(self, sample):
        raise NotImplementedError("Subclasses implement transformation (random, given, etc).")


class RandomTransformSE3(TransformSE3):
    def __init__(self, rot_mag: float = 45.0, trans_mag: float = 0.5, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        super().__init__()
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            rot_mag, trans_mag = np.random.uniform() * self._rot_mag, np.random.uniform() * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle /= np.linalg.norm(axis_angle)
        axis_angle *= np.deg2rad(rot_mag)
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = uniform_2_sphere()
        rand_trans *= np.random.uniform(high=trans_mag)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = transformed

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            sample['points_ref'] = np.random.permutation(sample['points_ref'])
            sample['points_src'] = np.random.permutation(sample['points_src'])
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


# Additional augmentations proposed in ReAgent

class Scale:
    """Scales source and target by a random scaling factor."""
    def __init__(self, scale=0.1, clip=0.5):
        self.scale = scale
        self.clip = clip

    def scale_points(self, points, scale):
        points_mean = points[:, :3].mean(axis=0)
        points[:, :3] = (points - points_mean) * scale + points_mean  # centroid location stays the same
        return points

    def __call__(self, sample: Dict):
        sample['scale'] = np.random.normal(1, self.scale, 3).clip(1 - self.clip, 1 + self.clip)

        sample['points_src'][:, :3] = self.scale_points(sample['points_src'][:, :3], sample['scale'])
        sample['points_ref'][:, :3] = self.scale_points(sample['points_ref'][:, :3], sample['scale'])
        sample['points_raw'][:, :3] = self.scale_points(sample['points_raw'][:, :3], sample['scale'])
        return sample


class Shear:
    """Shears source and target by a random angle and shear plane."""
    def __init__(self, scale=5, clip=15):
        self.scale = scale
        self.clip = clip

    def __call__(self, sample: Dict):
        angle = np.deg2rad(np.clip(np.random.normal(0, self.scale), -self.clip, self.clip))
        direction = uniform_2_sphere()
        normal = np.cross(direction, uniform_2_sphere())
        S = t3d.shears.sadn2mat(angle, direction, normal)

        sample['points_src'][:, :3] = (S @ sample['points_src'][:, :3].T).T
        sample['points_ref'][:, :3] = (S @ sample['points_ref'][:, :3].T).T
        sample['points_raw'][:, :3] = (S @ sample['points_raw'][:, :3].T).T

        return sample


class Mirror:
    """Mirrors source and target through a random plane of reflection."""
    def __call__(self, sample: Dict):
        normal = uniform_2_sphere()
        M = t3d.reflections.rfnorm2mat(normal)[:3, :3]

        sample['points_src'][:, :3] = (M @ sample['points_src'][:, :3].T).T
        sample['points_ref'][:, :3] = (M @ sample['points_ref'][:, :3].T).T
        sample['points_raw'][:, :3] = (M @ sample['points_raw'][:, :3].T).T

        return sample


class Normalize:
    """Normalizes source and target to be mean-centered and scales s.t. farthest point is of distance 1."""
    def __init__(self, using_target=True):
        self.using_target = using_target  # normalize wrt target

    def __call__(self, sample):
        t = sample['points_ref'][:, :3].mean(axis=0)  # center offset
        centered = sample['points_ref'][:, :3] - t
        dists = np.linalg.norm(centered, axis=1)
        s = dists.max()  # scale

        # apply to source and target
        sample['points_ref'][:, :3] = centered / s
        sample['points_src'][:, :3] = (sample['points_src'][:, :3] - t) / s

        # for test set with given estimate in unnormalized scale
        if 'transform_gt' in sample:
            sample['transform_gt'][:3, 3] /= s

        # keep track (to undo if needed)
        sample['normalization'] = np.eye(4, dtype=np.float32)
        sample['normalization'][np.diag_indices(3)] = s
        sample['normalization'][:3, 3] = t.squeeze()

        return sample


class GtTransformSE3(TransformSE3):
    """Takes transformation from GT dictionary and applies it to source/target for initial alignment."""
    def __init__(self, source_to_target=True):
        super().__init__()
        self.source_to_target = source_to_target

    def __call__(self, sample):
        if self.source_to_target:
            cam2model = np.eye(4, dtype=np.float32)
            cam2model[:3, :3] = sample['gt']['cam_R_m2c'].T
            cam2model[:3, 3] = -sample['gt']['cam_R_m2c'].T @ sample['gt']['cam_t_m2c'].squeeze()
            sample['points_src'], sample['est'], sample['c2m'] = self.apply_transform(sample['points_src'], cam2model)
        else:
            model2cam = np.eye(4, dtype=np.float32)
            model2cam[:3, :3] = sample['gt']['cam_R_m2c']
            model2cam[:3, 3] = sample['gt']['cam_t_m2c'].squeeze()
            sample['points_ref'], sample['c2m'], sample['est'] = self.apply_transform(sample['points_ref'], model2cam)
        return sample


class EstTransformSE3(TransformSE3):
    """Takes transformation from estimate and applies it to source/target for initial alignment."""
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        # this is our actual estimate -- it's residual from the best estimate is the initial pose error
        est = sample['est']
        inv_est = np.linalg.inv(est)  # model2cam -> cam2model -- this is then applied to src points (cam space)
        sample['points_src'], transform_r_s, transform_s_r = self.apply_transform(sample['points_src'], inv_est[:3, :])

        cam2model = np.eye(4, dtype=np.float32)
        cam2model[:3, :3] = sample['gt']['cam_R_m2c'].T
        cam2model[:3, 3] = -sample['gt']['cam_R_m2c'].T @ sample['gt']['cam_t_m2c'].squeeze()
        sample['transform_gt'] = (cam2model @ est)[:3, :]  # apply to source to get reference
        return sample


class SegmentResampler(Resampler):
    """Simulates imprecise segmentation by sampling nearest-neighbors to a random seed within the segmentation mask."""
    def __init__(self, num, p_fg=1.0, patch=True):
        super().__init__(num)
        self.p_fg = p_fg
        self.patch = patch  # sample continuous patch if true, else sample randomly in cloud

    def _patch_sample(self, fg_points, fg_size, bg_points, bg_size, K):
        # assuming points_src to be in camera space, we project to image space for kNN search
        def project(points):
            Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2]
            xs = np.divide(Xs * K[0, 0], Zs) + K[0, 2]
            ys = np.divide(Ys * K[1, 1], Zs) + K[1, 2]
            return np.hstack([xs[:, None], ys[:, None]])

        fg_2d = project(fg_points)

        # pick a random point on the object as center
        center_idx = np.random.randint(0, fg_points.shape[0])
        center_2d = fg_2d[center_idx]

        # find [fg_size] nearest points
        fg_centered = fg_2d - center_2d
        fg_distance = np.linalg.norm(fg_centered, axis=1)
        fg_neighbors = np.argsort(fg_distance)[:fg_size]
        fg = fg_points[fg_neighbors]

        if bg_size > 0:
            bg_2d = project(bg_points)
            # find [bg_size] nearest points
            bg_centered = bg_2d - center_2d
            bg_distance = np.linalg.norm(bg_centered, axis=1)
            bg_neighbors = np.argsort(bg_distance)[:bg_size]
            bg = bg_points[bg_neighbors]
        else:
            bg = []
            bg_neighbors = []

        return fg, bg, fg_neighbors, bg_neighbors

    def __call__(self, sample: Dict):
        # [p_fg]% from object, [1-p_fg]% from background
        fg = sample['points_src'][:, -1] > 0
        bg = sample['points_src'][:, -1] == 0
        if isinstance(self.p_fg, list):
            p_fg = np.random.uniform(*self.p_fg)
        else:
            p_fg = self.p_fg
        fg_size = math.ceil(p_fg * self.num)
        bg_size = math.floor((1 - p_fg) * self.num)
        assert fg_size + bg_size == self.num
        if self.patch:
            fg, bg, _, _ = self._patch_sample(sample['points_src'][fg], fg_size,
                                              sample['points_src'][bg], bg_size,
                                              sample['cam']['cam_K'])
        else:
            fg = self._resample(sample['points_src'][fg], fg_size)
            bg = self._resample(sample['points_src'][bg], bg_size)

        if fg_size > 0 and bg_size > 0:
            sample['points_src'] = np.vstack([fg, bg])
        elif fg_size > 0:
            sample['points_src'] = fg
        else:
            raise ValueError("only background pixels sampled")

        # num from model
        sample['points_ref'] = self._resample(sample['points_ref'], self.num)

        return sample
