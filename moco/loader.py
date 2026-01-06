# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional

from datasets import load_dataset
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


RVL_CDI_PHF_ID = "aharley/rvl_cdip"


class RvlCdipHFDataset(Dataset):
    """
    Hugging Face hosted RVL-CDIP dataset wrapper.
    Downloads/loads the split via `datasets` and returns a MoCo-friendly sample.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform=None,
        decode_rgb: bool = True,
    ) -> None:
        self.transform = transform
        self.decode_rgb = decode_rgb
        self.dataset = load_dataset(
            RVL_CDI_PHF_ID,
            split=split,
            cache_dir=cache_dir,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]

        # HF datasets returns PIL.Image.Image for Image feature
        if self.decode_rgb and img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # return a dummy label (unused for self-supervised learning)
        return img, 0
