import numpy as np
from torch.utils.data import Dataset
import torchvision
import os
import h5py
import pickle  # TODO or use h5py instead?
import trimesh

import config as cfg
import dataset.augmentation as Transforms

import sys
sys.path.append(cfg.BOP_PATH)
sys.path.append(os.path.join(cfg.BOP_PATH, "bop_toolkit_lib"))
import bop_toolkit_lib.inout as bop_inout
import bop_toolkit_lib.misc as bop_misc
import bop_toolkit_lib.dataset_params as bop_dataset_params


class DatasetModelnet40(Dataset):

    """Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet"""

    def __init__(self, split, noise_type):
        dataset_path = cfg.M40_PATH
        categories = np.arange(20) if split in ["train", "val"] else np.arange(20, 40)
        split = "test" if split == "val" else split  # ModelNet40 has no validation set - use cat 0-19 with test set

        self.samples, self.labels = self.get_samples(dataset_path, split, categories)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'points': self.samples[item, :, :], 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "clean":
            # 1-1 correspondence for each point (resample first before splitting), no noise
            if split == "train":
                transforms = [Transforms.Resampler(1024),
                              Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.FixedResampler(1024),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
        elif noise_type == "jitter":
            # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
            if split == "train":
                transforms = [Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(1024),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(1024),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for ModelNet40.")

        return torchvision.transforms.Compose(transforms)

    def get_samples(self, dataset_path, split, categories):
        filelist = [os.path.join(dataset_path, file.strip().split("/")[-1])
                   for file in open(os.path.join(dataset_path, f'{split}_files.txt'))]

        all_data = []
        all_labels = []
        for fi, fname in enumerate(filelist):
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels


class DatasetScanObjectNN(Dataset):

    def __init__(self, split, noise_type):
        dataset_path = cfg.SON_PATH
        split = "test" if split == "val" else split

        self.samples, self.labels = self.get_samples(dataset_path, split)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'points': self.samples[item, :, :], 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "sensor" and split == "test":
            transforms = [Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(),
                          Transforms.Resampler(2048),
                          Transforms.ShufflePoints()]
        else:
            raise ValueError(f"Only noise type 'sensor' supported for SceneObjectNN.")

        return torchvision.transforms.Compose(transforms)

    def get_samples(self, dataset_path, split):
        filelist = [os.path.join(dataset_path, "test_objectdataset.h5")]

        all_data = []
        all_labels = []
        for fi, fname in enumerate(filelist):
            f = h5py.File(fname, mode='r')
            data = f['data'][:].astype(np.float32)
            labels = f['label'][:].flatten().astype(np.int64)

            all_data.append(data)
            all_labels.append(labels)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels


class DatasetLinemod(Dataset):

    def __init__(self, split, noise_type):
        subsample = 16 if split == "eval" else 0  # only use every 16th test sample for evaluation during training
        split = "test" if split == "eval" else split
        self.samples, self.models = self.get_samples(split, subsample)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        model = self.models[item['obj_id']]

        # compose sample
        sample = {
            'idx': idx,
            'points_src': item['pcd'],
            'points_ref': model,
            'scene': item['scene'],
            'frame': item['frame'],
            'cam': item['cam'],
            'gt': item['gt'],
        }
        if 'est' in item:  # initial estimate only given for test split (using PoseCNN)
            sample['est'] = item['est']

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "segmentation":
            if split == "train":
                transforms = [
                    # resample segmentation (with [p_fg]% from object)
                    Transforms.SegmentResampler(1024, p_fg=[0.5, 1.0]),
                    # align source and target using GT -- easier to define error this way
                    Transforms.GtTransformSE3(),
                    # normalize source and target (mean centered, max dist 1.0)
                    Transforms.Normalize(),
                    # apply an initial pose error
                    Transforms.RandomTransformSE3(rot_mag=90.0, trans_mag=1.0, random_mag=True)
                ]
            elif split == "val":
                transforms = [
                    Transforms.SetDeterministic(),
                    Transforms.SegmentResampler(1024, p_fg=[0.5, 1.0]),
                    Transforms.GtTransformSE3(),
                    Transforms.Normalize(),
                    Transforms.RandomTransformSE3(rot_mag=90.0, trans_mag=1.0, random_mag=True)
                ]
            else:  # start from posecnn
                transforms = [
                    Transforms.SetDeterministic(),
                    # randomly resample inside segmentation mask (estimated by PoseCNN)
                    Transforms.SegmentResampler(1024, p_fg=1.0, patch=False),
                    # initial (erroneous) alignment using PoseCNN's pose estimation
                    Transforms.EstTransformSE3(),
                    Transforms.Normalize()
                ]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for LINEMOD.")
        return torchvision.transforms.Compose(transforms)

    def get_samples(self, split, subsample=0):
        model_params = bop_dataset_params.get_model_params('/'.join(cfg.LM_PATH.split('/')[:-1]),
                                                           cfg.LM_PATH.split('/')[-1], 'eval')
        mesh_ids = model_params['obj_ids']

        models = dict()
        for mesh_id in mesh_ids:
            mesh = trimesh.load(os.path.join(cfg.LM_PATH, f"models_eval/obj_{mesh_id:06d}.ply"))
            pcd, face_indices = trimesh.sample.sample_surface_even(mesh, 4096)
            models[mesh_id] = np.hstack([pcd, mesh.face_normals[face_indices]]).astype(np.float32)

        samples_path = f"reagent/{split}_posecnn.pkl" if split == "test" else f"reagent/{split}.pkl"
        with open(os.path.join(cfg.LM_PATH, samples_path), 'rb') as file:
            samples = pickle.load(file)
        if subsample > 0:  # used for evaluation during training
            samples = samples[::subsample]
        return samples, models

    # for visualization
    def get_rgb(self, scene_id, im_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"rgb/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_im(file_path)[..., :3]/255
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640, 3), dtype=np.float32)

    def get_depth(self, scene_id, im_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"depth/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_depth(file_path)
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640), dtype=np.float32)

    def get_seg(self, scene_id, im_id, gt_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"mask_visib/{im_id:06d}_{gt_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_im(file_path)
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640), dtype=np.uint8)
