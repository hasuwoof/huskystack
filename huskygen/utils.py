from __future__ import annotations

import os
from typing import Optional

import inflect

from huskygen.constants import inflectable_tag_re

inflect_engine = inflect.engine()


def format_human_readable(tag: str):
    if inflectable_tag_re.match(tag) is not None:
        try:
            split_tag = tag.split('_')
            split_tag[0] = inflect_engine.number_to_words(split_tag[0])
            tag = '_'.join(split_tag)
        except Exception as e:
            print(e)
    return tag.replace('_', ' ')


def get_latest_post_file(base_dir: str) -> Optional[str]:
    filename = None
    for file in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.startswith('posts-') and file.endswith('.gz'):
            filename = os.path.join(base_dir, file)

    return filename


def get_latest_tags_file(base_dir: str) -> Optional[str]:
    filename = None
    for file in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.startswith('tags-') and file.endswith('.gz'):
            filename = os.path.join(base_dir, file)

    return filename


def get_latest_implications(base_dir: str) -> Optional[str]:
    filename = None
    for file in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.startswith('tag_implications-') and file.endswith('.gz'):
            filename = os.path.join(base_dir, file)

    return filename
