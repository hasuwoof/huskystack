import io
import json
import os
from multiprocessing import Pool
from os.path import abspath
from typing import Dict, Optional
from urllib import request

import PIL
import tqdm
from PIL.Image import Image
from minio import Minio

from huskydb.models import ImageTag, ImageEntry
from huskydb.store import HuskyStore

TARGET_SMALLEST_SIDE = 1024
MAX_LARGEST_SIDE = 2048
UPLOAD_THREADS = 32


def get_reduced_size(post: dict):
    w, h = post['w'], post['h']

    max_side = max(w, h)
    min_side = min(w, h)

    target_ratio = TARGET_SMALLEST_SIDE / min_side
    max_ratio = MAX_LARGEST_SIDE / max_side

    if target_ratio > 1:
        return w, h

    ratio = min(target_ratio, max_ratio)

    return round(w * ratio), round(h * ratio)


def get_filename(post: dict):
    return f"{post['id']}.{post['ext']}"


def build_dataset():
    cwd = os.getcwd()
    metadata_dir = os.path.join(cwd, 'parsed_data')

    db_file_path = abspath('./data/e6.sqlite')
    if os.path.isfile(db_file_path):
        print(f'File exists, skipping dataset construction: {db_file_path}')
        return

    print('Loading tags...')

    with open(os.path.join(metadata_dir, 'tags.json'), 'r') as f:
        tags: Dict[str, int] = json.load(f)
        sorted_tags = sorted([(tag_type, tag) for tag, tag_type in tags.items()])

    ds = HuskyStore(abspath('./data'), disable_foreign_key=True)
    ds.init()

    print('Adding tags...')

    ds.db.clear()
    ds.db.add_tags([
        ImageTag(tag=tag, type=tag_type) for tag_type, tag in sorted_tags
    ])

    print(f'Added {len(tags)} tags')

    print('Loading dataset...')
    with open(os.path.join(metadata_dir, 'meta.jsonl'), 'rt') as f:
        entries = [json.loads(s) for s in f]

    print(f'Dataset with {len(entries)} entries loaded')

    cursor = ds.db.get_cursor()

    for post in tqdm.tqdm(entries, desc='Adding posts'):
        w, h = get_reduced_size(post)
        filename = get_filename(post)
        raw_tags: dict = post['tags']

        post_tags = set()
        for k, value in raw_tags.items():
            if k == 'rating':
                post_tags.add(value)
            else:
                post_tags = post_tags.union(value)

        ds.db.store_entry(ImageEntry.parse_obj({
            'filename': filename,
            'original_width': post['w'],
            'original_height': post['h'],
            'width': w,
            'height': h,
            'metadata': post
        }), tags=list(post_tags), cursor=cursor)

    ds.db.commit()

    print('Running VACUUM')
    ds.db.db.execute('VACUUM')

    ds.close()


def get_image_from_url(url: str) -> Optional[Image]:
    try:
        image = PIL.Image.open(io.BytesIO(request.urlopen(url).read())).convert('RGB')
        if image.width > 0 and image.height:
            return image
    except Exception:
        pass

    return None


_image_processor_minio_instance = None


class ImageProcessor:
    ext_alias = {
        'jpg': 'jpeg'
    }

    def __init__(self):
        self.image_object_prefix = 'img/'
        self.key_id = os.getenv('S3_KEY_ID')
        self.key_secret = os.getenv('S3_KEY_SECRET')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.s3_backend = os.getenv('S3_BACKEND_URL')

    def process_entry(self, entry: ImageEntry):
        try:
            image = get_image_from_url(entry.metadata['url'][0])
            if entry.width != entry.original_width or entry.height != entry.original_height:
                image_resized = image.resize((entry.width, entry.height))
            else:
                image_resized = image

            client = self.get_minio()
            in_mem_file = io.BytesIO()
            original_ext = entry.metadata['ext']
            ext = self.ext_alias[original_ext] if original_ext in self.ext_alias else original_ext
            image_resized.save(in_mem_file, format=ext)
            in_mem_file.seek(0)

            client.put_object(self.bucket_name, f'{self.image_object_prefix}{entry.filename}', in_mem_file,
                              in_mem_file.getbuffer().nbytes, content_type='image/png')

        except Exception as e:
            return entry, f'Failed with exception: {e.__class__.__name__}: {e}'

        return entry, None

    def get_minio(self):
        global _image_processor_minio_instance

        if _image_processor_minio_instance:
            return _image_processor_minio_instance

        print('minio cache miss')

        s3_client = Minio(self.s3_backend,
                          access_key=self.key_id,
                          secret_key=self.key_secret,
                          secure=True)

        _image_processor_minio_instance = s3_client

        return s3_client


def upload_images():
    ds = HuskyStore(abspath('./data'))
    ds.init()

    upload_progress = set()
    try:
        with open(abspath('./data/upload_progress.txt'), 'r') as f:
            upload_progress = set(e.strip() for e in f.readlines())
    except OSError:
        pass

    entries_to_process = []
    for entry in tqdm.tqdm(ds.entries_iterator(), desc='Loading entries from db...', total=ds.get_image_count()):
        if entry.filename not in upload_progress:
            entries_to_process.append(entry)

    print(f'{len(upload_progress)} already processed, {len(entries_to_process)} to go')

    processor = ImageProcessor()

    with Pool(UPLOAD_THREADS) as pool:
        result_iter = pool.imap_unordered(processor.process_entry, entries_to_process)

        for entry, result in tqdm.tqdm(result_iter, desc='Resizing and uploading', total=len(entries_to_process)):
            entry: ImageEntry
            if result:
                print(f'Entry {entry.id} failed: {result}')
            else:
                with open(abspath('./data/upload_progress.txt'), 'a') as f:
                    f.write(f'{entry.filename}\n')

    ds.close()


if __name__ == '__main__':
    build_dataset()
    upload_images()
