from typing import Iterator

from torch.utils.data import Sampler

from huskyloader.bucket import HuskyBucket


class HuskyBatchSampler(Sampler):
    """
    A sampler that returns samples from a HuskyBucket
    """
    def __init__(self, bucket: HuskyBucket):
        super().__init__(None)
        self.bucket = bucket

    def __iter__(self):
        yield from self.bucket.get_batch_iterator()

    def __len__(self):
        return self.bucket.get_batch_count()
