"""Memory-mapped official-training tensors with explicit decoding and train-only augmentation."""

import hashlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from damage_analytics.task07.data import make_loader, prepare_image
from damage_analytics.task10.config import TrainConfig
from damage_analytics.task10.encoding import rgb_from_array
from damage_analytics.task10.release import Release


class SeverityDataset(Dataset):
    def __init__(
        self,
        release: Release,
        split: str,
        settings: TrainConfig,
        seed: int,
        *,
        allow_inferred_encoding: bool = False,
    ) -> None:
        self.rows = release.samples(split)
        if not self.rows:
            raise ValueError("Requested development partition is empty")
        if (
            not release.metadata["inputs"]["settings"]["encoding_verified"]
            and not allow_inferred_encoding
        ):
            raise ValueError(
                "Publisher color encoding is unverified; "
                "explicitly acknowledge provisional decoding"
            )
        self.release, self.training, self.settings, self.seed = (
            release,
            split == "train",
            settings,
            seed,
        )
        self.epoch = torch.tensor(0, dtype=torch.int64)
        if settings.num_workers:
            self.epoch.share_memory_()
        self._x, self._y = None, None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_x"], state["_y"] = None, None  # Spawn workers reopen maps; never pickle GB arrays.
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, value: int) -> None:
        self.epoch.fill_(value)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        if row["official_partition"] != "train":
            raise ValueError("Official test cannot enter a development Dataset")
        if self._x is None:
            self._x = np.load(
                self.release.file(row["image_array"], verify=False),
                mmap_mode="r",
                allow_pickle=False,
            )
            self._y = np.load(
                self.release.file(row["label_array"], verify=False),
                mmap_mode="r",
                allow_pickle=False,
            )
        values, label = self._x[row["array_index"]], self._y[row["array_index"]]
        if (
            hashlib.sha256(values.tobytes()).hexdigest() != row["raw_sha256"]
            or hashlib.sha256(label.tobytes()).hexdigest() != row["label_sha256"]
            or not np.isfinite(label).all()
            or not np.all((label == 0) | (label == 1))
            or label.sum() != 1
            or int(label.argmax()) != row["class_id"]
        ):
            raise ValueError("Tensor row/label changed since preparation")
        rgb = rgb_from_array(
            values, self.release.metadata["inputs"]["settings"]["roundtrip_tolerance"]
        )
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self.epoch), index]))
        prepared = prepare_image(
            Image.fromarray(rgb), self.settings.image_size, training=self.training, rng=rng
        )
        return torch.from_numpy(prepared), row["class_id"], row["sample_id"]


def loader(
    release: Release, split: str, settings: TrainConfig, seed: int, *, provisional: bool = False
):
    dataset = SeverityDataset(release, split, settings, seed, allow_inferred_encoding=provisional)
    return make_loader(
        dataset, settings.batch_size, settings.num_workers, seed, shuffle=split == "train"
    )
