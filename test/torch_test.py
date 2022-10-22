import os
from os.path import abspath

import torch
import numpy as np
import random
import pprint

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from huskydb.store import HuskyStore
from huskyloader.bucket import HuskyBucket
from huskyloader.dataset import HuskyDataset
from huskyloader.sampler import HuskyBatchSampler

if __name__ == "__main__":
    batch_size = 4

    data_path = abspath(os.path.join(os.path.dirname(__file__), '../data'))

    husky_dataset = HuskyDataset(data_path)

    bucket = HuskyBucket(data_path, num_buckets=101, batch_size=4, max_ratio=2, max_entries=1000)

    custom_sampler = HuskyBatchSampler(bucket)

    data_loader_sampler = torch.utils.data.DataLoader(
        husky_dataset,
        num_workers=4,
        batch_sampler=custom_sampler
    )

    for i in range(0, 8):
        print(f"Step {i}")
        for batch in tqdm(data_loader_sampler):
            # print(batch.shape)
            pass
