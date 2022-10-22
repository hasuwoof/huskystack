import itertools
import random
from typing import Dict, List, Generator, Tuple, Optional

import numpy as np
import tqdm
from scipy.interpolate import interp1d

from huskydb.models import ImageEntry
from huskydb.store import HuskyStore
from huskyloader.dataset import HuskyDataset


class HuskyBucket:
    def __init__(self, data_path: str,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 768,
                 bucket_side_increment: int = 64,
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 2,
                 max_entries: Optional[int] = None):
        """

        :param dataset:
        :param num_buckets:
        :param batch_size: The size of the batch.
                           Buckets will have an element count that is a multiple of the batch size
        :param bucket_side_min:
        :param bucket_side_max:
        :param bucket_side_increment:
        :param max_image_area:
        :param max_ratio:
        """

        if bucket_side_increment <= 0:
            raise ValueError('bucket_increment should be a positive number')

        if bucket_side_min % bucket_side_increment != 0:
            raise ValueError('bucket_length_min must be a multiple of bucket_increment')

        if bucket_side_max % bucket_side_increment != 0:
            raise ValueError('bucket_length_max must be a multiple of bucket_increment')

        if num_buckets % 2 == 0:
            raise ValueError('num_buckets must be odd')

        self.store = HuskyStore(data_path)
        self.store.init()

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0
        self.max_entries = max_entries

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()

    def init_buckets(self):
        # calculate all the possible side lengths our bucket can have using the increment as steps
        possible_lengths = list(range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))

        # generate all possible buckets that satisfy the max_ratio and max_image_area criteria
        # for now we'll only include landscape buckets (where ratio > 1), the portrait buckets will be symmetrically
        # constructed later
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                                if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)

        buckets_by_ratio = {}

        # group the buckets by their aspect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=_sort_by_area)[-1]
                                       for buckets in buckets_by_ratio.values()], key=_sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {*np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=_sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)

        for b in buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int], List[int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            ((w, h), [image1, image2, ..., image{batch_size}])

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator(include_all=False, max_entries=self.max_entries)
        total_dropped = 0
        if self.max_entries is None:
            total_entries = self.store.get_image_count()
        else:
            total_entries = min(self.store.get_image_count(), self.max_entries)

        for entry in tqdm.tqdm(entries, total=total_entries, desc='Filling buckets'):
            if entry is None or not self._process_entry(entry):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: ImageEntry):
        w = entry.width
        h = entry.height
        aspect = w / h

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            # bigger than the max aspect ratio
            return False

        # if not self.dataset.include(entry):
        #     return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket_index = round(float(best_bucket))
        bucket = self.buckets[bucket_index]

        self.bucket_data[bucket].append(entry.id)

        return True


def _sort_by_area(bucket: tuple):
    w, h = bucket
    return w * h


def _sort_by_ratio(bucket: tuple):
    w, h = bucket
    return w / h
