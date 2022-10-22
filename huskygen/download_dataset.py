import glob
import os
import urllib.request
from itertools import groupby
from multiprocessing.pool import ThreadPool

import htmllistparse
from tqdm import tqdm

DATADUMP_FOLDER = './e621_data'
DB_EXPORT_BASEULR = 'https://e621.net/db_export/'

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent',
                      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36')]
urllib.request.install_opener(opener)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def ensure_directory_exists():
    if not os.path.exists(DATADUMP_FOLDER):
        os.makedirs(DATADUMP_FOLDER)


def remove_old_files():
    files = glob.glob(f'{DATADUMP_FOLDER}/.*')
    for f in files:
        os.remove(f)


def fetch_dataset_meta():
    cwd, listing = htmllistparse.fetch_listing(DB_EXPORT_BASEULR, timeout=30)
    filtered_list = list()

    for key, group in groupby(listing,
                              lambda x: x.name.split('-', 1)[0]):  # groups based on the start 10 characters of file
        filtered_list.append([item for item in group][-1])

    return [x.name for x in filtered_list]


def get_file_from_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_file_from_dump(filename):
    url = f'{DB_EXPORT_BASEULR}{filename}'
    get_file_from_url(url, f'{DATADUMP_FOLDER}/{filename}')


def fetch_dataset():
    files = ['posts', 'tags', 'tag_aliases', 'tag_implications']

    file_list = fetch_dataset_meta()

    for file in file_list:
        if len(list(filter(file.startswith, files))) > 0:
            get_file_from_dump(file)
