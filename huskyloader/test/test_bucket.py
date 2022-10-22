import functools
import os.path
import unittest
from os.path import abspath
from tempfile import TemporaryDirectory

from huskydb.store import HuskyStore
from huskyloader.bucket import HuskyBucket
from huskyloader.dataset import HuskyDataset


def with_temp_store(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'husky.yml'), 'w') as f:
                f.write("""dataset_name: temp\nimage_dir: ./img\nindex_dir: .""")

            with HuskyStore(abspath(tempdir)) as store:
                return func(*args, store=store, **kwargs)

    return wrapped


class BucketTests(unittest.TestCase):
    @with_temp_store
    def test_sample_data(self, store: HuskyStore):
        expected_total_images = 1000

        # using random data in tests is usually a bad idea
        store.db.fill_with_mock_data(expected_total_images)

        bucket = HuskyBucket(HuskyDataset(store), 11, batch_size=4, max_ratio=2)

        all_images = set()
        total_batches = 0
        for batch in bucket.get_batch_iterator():
            bucket_info, entries = batch
            total_batches += 1
            for image in entries:
                self.assertNotIn(image, all_images)

                all_images.add(image)

        self.assertEqual(total_batches, bucket.get_batch_count())
        self.assertEqual(expected_total_images, len(all_images) + bucket.total_dropped)

    def test_local_data(self):
        dirname = os.path.dirname(__file__)
        with HuskyStore(abspath(os.path.join(dirname, '../../data'))) as store:
            bucket = HuskyBucket(HuskyDataset(store), num_buckets=101, batch_size=4, max_ratio=2)

            all_images = set()
            total_batches = 0
            for batch in bucket.get_batch_iterator():
                bucket_info, entries = batch
                total_batches += 1
                for image in entries:
                    self.assertNotIn(image, all_images)

                    all_images.add(image)
