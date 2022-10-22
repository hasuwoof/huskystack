import os.path
from typing import BinaryIO
from urllib.parse import urljoin

import urllib3 as urllib3


class HuskyDataSource:
    def __init__(self, base_dir: str, *, base_url: str, extensive_validation: bool = False,
                 available_files_index: str = None):
        self.pool = urllib3.PoolManager()
        self.base_url = base_url
        self.extensive_validation = extensive_validation

        if available_files_index:
            available_files_path = os.path.abspath(os.path.join(base_dir, available_files_index))
            with open(available_files_path, 'r') as f:
                self.available_files = set([line.strip() for line in f.readlines() if line.strip()])
        else:
            self.available_files = None

    def download_index_file(self, f: BinaryIO, filename: str):
        pass

    def download_image_file(self, f: BinaryIO, filename: str):
        r = self.pool.urlopen('GET', self.get_object_url(filename))
        f.write(r.data)

    def is_file_available(self, filename: str):
        return self.available_files is None or filename in self.available_files

    def close(self):
        self.pool.clear()

    def get_object_url(self, filename: str):
        return urljoin(self.base_url, filename)

    def __getstate__(self):
        return {**self.__dict__, 'pool': None}

    def __setstate__(self, state):
        self.__dict__ = state
        self.pool = urllib3.PoolManager()
