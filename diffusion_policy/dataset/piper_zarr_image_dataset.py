from typing import Dict, List, Optional
import copy
import os

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class PiperZarrImageDataset(BaseImageDataset):
    """Dataset for preprocessed Piper zarr episodes.

    Images are already resized to the training resolution and stored as uint8,
    so training avoids HDF5 random reads and per-sample resize work.
    """

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon: int = 16,
            pad_before: int = 0,
            pad_after: int = 0,
            n_obs_steps: Optional[int] = None,
            n_latency_steps: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            image_obs_map: Optional[Dict[str, str]] = None,
            lowdim_obs_map: Optional[Dict[str, str]] = None,
            action_key: str = "action",
            valid_key: str = "valid",
            episode_paths: Optional[List[str]] = None,
            episode_mask: Optional[np.ndarray] = None,
        ):
        dataset_path = os.path.expanduser(dataset_path)
        assert os.path.isdir(dataset_path), dataset_path

        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.action_key = action_key
        self.valid_key = valid_key

        obs_meta = shape_meta["obs"]
        self.rgb_keys = [
            key for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            key for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        ]
        self.action_shape = tuple(shape_meta["action"]["shape"])

        self.image_obs_map = {
            "camera_wrist": "camera_wrist",
            "camera_global": "camera_global",
        }
        if image_obs_map is not None:
            self.image_obs_map.update(image_obs_map)
        self.lowdim_obs_map = {
            "robot_qpos": "robot_qpos",
            "robot_eef_pose": "robot_eef_pose",
        }
        if lowdim_obs_map is not None:
            self.lowdim_obs_map.update(lowdim_obs_map)

        if episode_paths is None:
            episode_paths = sorted(
                os.path.join(dataset_path, x)
                for x in os.listdir(dataset_path)
                if x.endswith(".zarr")
            )
        assert len(episode_paths) > 0, f"No .zarr episodes found in {dataset_path}"
        self.episode_paths = list(episode_paths)

        self.episode_lengths = self._read_episode_lengths(self.episode_paths, self.action_key)
        if episode_mask is None:
            episode_mask = self._make_train_episode_mask(
                n_episodes=len(self.episode_paths),
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                seed=seed,
            )
        self.episode_mask = np.asarray(episode_mask, dtype=bool)
        self.val_mask = ~self.episode_mask

        self.indices = self._build_indices()
        assert len(self.indices) > 0, (
            "No valid training windows found. Check valid masks, horizon, "
            "n_latency_steps, and episode lengths."
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.episode_mask = self.val_mask.copy()
        val_set.val_mask = ~val_set.episode_mask
        val_set.indices = val_set._build_indices()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        stats = self._compute_lowdim_stats()

        normalizer["action"] = get_range_normalizer_from_stat(stats["action"])
        for key in self.lowdim_keys:
            normalizer[key] = get_range_normalizer_from_stat(stats[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = list()
        for episode_idx, path in enumerate(self.episode_paths):
            if not self.episode_mask[episode_idx]:
                continue
            root = zarr.open(path, mode="r")
            valid = root[self.valid_key][:].astype(bool)
            action = root[self.action_key][:]
            action = action[valid]
            if len(action) > 0:
                actions.append(action.astype(np.float32))
        if len(actions) == 0:
            return torch.empty((0,) + self.action_shape, dtype=torch.float32)
        return torch.from_numpy(np.concatenate(actions, axis=0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        threadpool_limits(1)
        episode_idx, start_idx = self.indices[idx]
        path = self.episode_paths[episode_idx]
        sequence_length = self.horizon + self.n_latency_steps
        end_idx = start_idx + sequence_length
        obs_end = start_idx + (self.n_obs_steps or sequence_length)

        root = zarr.open(path, mode="r")
        obs_dict = dict()
        for key in self.rgb_keys:
            zarr_key = self.image_obs_map[key]
            imgs = root[zarr_key][start_idx:obs_end]
            imgs = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = imgs

        for key in self.lowdim_keys:
            zarr_key = self.lowdim_obs_map[key]
            obs_dict[key] = root[zarr_key][start_idx:obs_end].astype(np.float32)

        action = root[self.action_key][start_idx:end_idx].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        return {
            "obs": {
                key: torch.from_numpy(value)
                for key, value in obs_dict.items()
            },
            "action": torch.from_numpy(action),
        }

    @staticmethod
    def _read_episode_lengths(episode_paths: List[str], action_key: str) -> np.ndarray:
        lengths = list()
        for path in episode_paths:
            root = zarr.open(path, mode="r")
            lengths.append(int(root[action_key].shape[0]))
        return np.asarray(lengths, dtype=np.int64)

    @staticmethod
    def _make_train_episode_mask(
            n_episodes: int,
            val_ratio: float,
            max_train_episodes: Optional[int],
            seed: int,
        ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        val_mask = np.zeros(n_episodes, dtype=bool)
        if val_ratio > 0:
            n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
            val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
            val_mask[val_idxs] = True
        train_mask = ~val_mask
        if max_train_episodes is not None and np.sum(train_mask) > max_train_episodes:
            train_idxs = np.nonzero(train_mask)[0]
            keep_idxs = rng.choice(train_idxs, size=int(max_train_episodes), replace=False)
            train_mask = np.zeros(n_episodes, dtype=bool)
            train_mask[keep_idxs] = True
        return train_mask

    def _build_indices(self) -> np.ndarray:
        sequence_length = self.horizon + self.n_latency_steps
        indices = list()
        for episode_idx, path in enumerate(self.episode_paths):
            if not self.episode_mask[episode_idx]:
                continue
            root = zarr.open(path, mode="r")
            valid = root[self.valid_key][:].astype(bool)
            T = len(valid)
            if T < sequence_length:
                continue
            window = np.ones(sequence_length, dtype=np.int64)
            valid_window_counts = np.convolve(valid.astype(np.int64), window, mode="valid")
            valid_starts = np.nonzero(valid_window_counts == sequence_length)[0]
            for start_idx in valid_starts:
                indices.append((episode_idx, int(start_idx)))
        return np.asarray(indices, dtype=np.int64)

    def _compute_lowdim_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        accum = {"action": list()}
        for key in self.lowdim_keys:
            accum[key] = list()

        for episode_idx, path in enumerate(self.episode_paths):
            if not self.episode_mask[episode_idx]:
                continue
            root = zarr.open(path, mode="r")
            valid = root[self.valid_key][:].astype(bool)
            if not np.any(valid):
                continue
            accum["action"].append(root[self.action_key][:][valid].astype(np.float32))
            for key in self.lowdim_keys:
                zarr_key = self.lowdim_obs_map[key]
                accum[key].append(root[zarr_key][:][valid].astype(np.float32))

        stats = dict()
        for key, values in accum.items():
            assert len(values) > 0, f"No valid values found for {key}"
            stats[key] = array_to_stats(np.concatenate(values, axis=0))
        return stats
