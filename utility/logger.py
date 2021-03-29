import os
import glob
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Based off https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/logger.py"""
    def __init__(self, log_dir, log_name, reset_num_timesteps=True):
        self.name_to_value = defaultdict(float)  # values this iteration
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)

        latest_run_id = self.get_latest_run_id(log_dir, log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(log_dir, f"{log_name}_{latest_run_id + 1}")
        os.makedirs(save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=save_path)

    @staticmethod
    def get_latest_run_id(log_dir, log_name) -> int:
        """
        Returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.
        :return: latest run number
        """
        max_run_id = 0
        for path in glob.glob(f"{log_dir}/{log_name}_[0-9]*"):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def record(self, key, value, exclude=None):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.
        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = exclude

    def dump(self, step=0):
        """Write all of the diagnostics from the current iteration"""
        self.write(self.name_to_value, self.name_to_excluded, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def write(self, key_values, key_excluded, step=0):

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if isinstance(value, np.ScalarType):
                self.writer.add_scalar(key, value, step)

            if isinstance(value, torch.Tensor):
                self.writer.add_histogram(key, value, step)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """Closes the file"""
        if self.writer:
            self.writer.close()
            self.writer = None
