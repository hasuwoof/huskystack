import os
from multiprocessing.pool import ThreadPool
from os import path
from typing import Optional, Generator, Tuple

import tqdm
import yaml
from PIL import Image

from huskydb.database import HuskyDatabase
from huskydb.datasource import HuskyDataSource
from huskydb.models import HuskyConfig, ImageEntry


class HuskyStore:
    def __init__(self, data_dir: str, disable_foreign_key=False):
        super().__init__()

        self.dir = data_dir
        self.config: HuskyConfig = self.load_config()
        self.image_dir = path.abspath(path.join(self.dir, self.config.image_dir))
        self.index_dir = path.abspath(path.join(self.dir, self.config.index_dir))
        self.index_file = path.abspath(path.join(self.index_dir, self.config.dataset_name + '.sqlite'))

        self.db: HuskyDatabase = HuskyDatabase(index_file=self.index_file, disable_foreign_key=disable_foreign_key)
        self.ds: Optional[HuskyDataSource] = HuskyDataSource(self.dir, **self.config.data_source) \
            if self.config.data_source else None

    def load_config(self) -> HuskyConfig:
        config_path = path.join(self.dir, 'husky.yml')

        if not path.isfile(config_path):
            raise OSError(f'Cannot find config file in {self.dir}')

        with open(config_path, encoding='utf-8') as f:
            return HuskyConfig.parse_obj(yaml.full_load(f))

    def soft_init(self):
        self.db.init()

    def init(self):
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        self.db.init()

    def get_image_path(self, filename: str):
        return path.abspath(path.join(self.image_dir, filename))

    def download_all(self, threads: int = 0, use_tqdm: bool = False):
        available_files = [(i, f) for i, f in self.db.get_filenames() if self.ds.is_file_available(f)]
        value_iter = available_files

        if threads > 0:
            pool = ThreadPool(threads)
            value_iter = pool.imap_unordered(self.download_file, value_iter)
        else:
            pool = None
            value_iter = map(self.download_file, value_iter)

        if use_tqdm:
            value_iter = tqdm.tqdm(value_iter, total=len(available_files), desc='Downloading dataset')

        for item_id, file_path, available in value_iter:
            if not available:
                print(f"Item {item_id} ({file_path}) was downloaded, but failed validation and was deleted")

        if pool:
            pool.close()

    def download_file(self, id_file: Tuple[int, str]):
        item_id, file = id_file
        file_path = self.get_image_path(file)

        if not path.isfile(file_path):
            with open(file_path, 'wb') as f:
                self.ds.download_image_file(f, file)

        if self.config.extended_validation_when_downloading:
            available = self.perform_extended_validation(item_id, file_path)
        else:
            available = True

        return item_id, file_path, available

    def check_item_exists(self, item: ImageEntry):
        file_path = self.get_image_path(item.filename)
        return path.isfile(file_path)

    def close(self):
        self.db.close()

        if self.ds:
            self.ds.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __enter__(self):
        self.init()
        return self

    def entries_iterator(self, include_all=True, max_entries=None) -> Generator[ImageEntry, None, None]:
        if include_all:
            yield from self.db.get_entries_iterator()
        else:
            all_files_in_dir = set(os.listdir(self.image_dir))
            generated_entries = 0
            # TODO: should probably store the ids instead so we don't have to iterate over the whole db?
            for entry in self.db.get_entries_iterator():
                if generated_entries == max_entries:
                    break

                if entry.filename in all_files_in_dir:
                    generated_entries += 1
                    yield entry
                else:
                    # yield to advance the progress
                    yield None

    def get_image_count(self):
        return self.db.get_entry_count()

    def get_entry_by_id(self, idx: int, fetch_relationships: bool = True) -> ImageEntry:
        if fetch_relationships:
            return self.db.get_entry(idx)
        else:
            return self.db.get_entry_no_extra(idx)

    def perform_extended_validation(self, item_id: int, file_path: str):
        entry = self.get_entry_by_id(item_id, fetch_relationships=False)

        # noinspection PyBroadException
        try:
            image = Image.open(file_path)
        except Exception:
            return False

        if image.width != entry.width or image.height != entry.height:
            return False

        return True
