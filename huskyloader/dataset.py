import itertools
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from huskydb.models import ImageEntry
from huskydb.store import HuskyStore


class HuskyDataset(Dataset):
    def __init__(self,
                 store_path: str = './data',
                 flip_probability: float = 0.5,
                 shuffle_tags: bool = True,
                 ucg_percentage: float = 0.0):
        self.rng = random.Random()

        self.store = HuskyStore(store_path)

        self.flip_probability = flip_probability

        self.shuffle_tags = shuffle_tags
        self.ucg_percentage = ucg_percentage

        self.store.soft_init()

        self.flip = transforms.RandomHorizontalFlip(p=flip_probability)

        print(f"Store has {self.store.get_image_count()} images")

    def __getitem__(self, image_spec: Tuple[int, int, int]):
        return_dict = {"image": None, "caption": ""}

        is_ucg = False
        image_id, w, h = image_spec
        if self.ucg_percentage > 0:
            if np.random.random() < self.ucg_percentage:
                is_ucg = True

        entry = self.store.get_entry_by_id(image_id, not is_ucg)

        try:
            return_dict['image'] = self.get_image(entry, w, h)
        except (OSError, ValueError) as e:
            print(f"Error loading image {entry.filename}: {e}")
            return_dict['image'] = np.random.rand(3, 255, 255)
            is_ucg = True

        if not is_ucg:
            return_dict['caption'] = self.get_caption(entry)

        return return_dict

    def get_image(self, entry: ImageEntry, w: int, h: int) -> np.ndarray:
        entry_path = self.store.get_image_path(entry.filename)
        image = Image.open(entry_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = ImageOps.fit(image, (w, h), bleed=0.0, centering=(0.5, 0.5), method=Image.Resampling.LANCZOS)
        image = self.flip(image)
        image_np = np.array(image).astype(np.uint8)

        # XXX: If you want to use the dataset with the new trainer, transpose (or rollaxis(image_np, -1, 0))
        return (image_np / 127.5 - 1.0).astype(np.float32)#.transpose(2, 0, 1)

    def get_caption(self, entry: ImageEntry) -> str:
        # if self.shuffle_tags:
        #    self.rng.shuffle(entry.tags)

        # TODO: Implement tag priority + sprinking in leftover tags
        return " ".join(itertools.chain(*entry.tags.values()))

    def include(self, entry: ImageEntry):
        # TODO: Implement runtime filtering here
        return True

    def __len__(self):
        return self.store.get_image_count()
