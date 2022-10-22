from __future__ import annotations
from typing import Dict, Generator

import csv
import gzip
import itertools
import json
import os.path
import sys
import networkx as nx

from tqdm import tqdm

from huskygen.constants import allowed_extensions, bad_tags, good_tags, tag_key_to_type
from huskygen.utils import get_latest_post_file, get_latest_tags_file, get_latest_implications
from huskygen.filter.datatypes import SortedList
from huskygen.models import PostData, TagMeta, TagCategory, PostRating

max_int = sys.maxsize

while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        print('Too big, we dividing again.')
        max_int = int(max_int / 10)


def get_posts(filename: str, max_len: int = -1) -> Generator[PostData]:
    index = 0

    with gzip.open(filename, 'rt', encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['is_deleted'] == 't':
                continue

            md5 = row['md5']
            tags = row['tag_string'].split(' ')
            urls = [f'https://static1.e621.net/data/{md5[:2]}/{md5[2:4]}/{md5}.{row["file_ext"]}']
            # if 'hi_res' in tags:
            #     urls.append(f'https://static1.e621.net/data/sample/{md5[:2]}/{md5[2:4]}/{md5}.jpg')

            if max_len != -1:
                index += 1
                if index > max_len:
                    return

            yield PostData(
                id=row['id'],
                created_at=row['created_at'],
                md5=row['md5'],
                url=urls,
                rating=PostRating(row['rating']),
                width=int(row['image_width']),
                height=int(row['image_height']),
                tags=tags,
                fav_count=int(row['fav_count']),
                score=int(row['score']),
                comment_count=int(row['comment_count']),
                file_ext=row['file_ext'],
                file_size=int(row['file_size']),
                duration=None if row['duration'] == '' else float(row['duration']),
            )


def generate_tag_mapping(filename: str) -> Dict[str, TagMeta]:
    tags = {}

    with gzip.open(filename, 'rt', encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags[row['name']] = TagMeta(
                id=row['id'],
                name=row['name'],
                category=TagCategory(int(row['category'])),
                post_count=int(row['post_count'])
            )

    return tags


def generate_tag_implication_tree(filename: str) -> set:
    tags = set()

    with gzip.open(filename, 'rt', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            antecedent = row['antecedent_name']
            consequent = row['consequent_name']
            tags.add(antecedent)

    return tags


def generate_tag_implication_graph(filename: str, tags_meta: Dict[str, TagMeta]) -> nx.DiGraph:
    impls = []

    with gzip.open(filename, 'rt', encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            antecedent = row['antecedent_name']
            consequent = row['consequent_name']
            status = row['status']
            if (status == 'active'
                    and antecedent in tags_meta.keys()
                    and consequent in tags_meta.keys()
                    and tags_meta[antecedent].post_count > 0
                    and tags_meta[consequent].post_count > 0
                    and tags_meta[antecedent].category == TagCategory.SPECIES
                    and tags_meta[consequent].category == TagCategory.SPECIES):
                impls.append((antecedent, consequent))

    graph = nx.DiGraph()
    graph.add_edges_from(impls)

    return graph


def is_good_shit(post: PostData):
    for tag in post.tags:
        if tag in good_tags:
            return True

    return False


def update_seen_tags(tag_types: dict, tags: dict):
    for g, tag_list in tags.items():
        tag_type = tag_key_to_type[g]

        if g == 'rating':
            tag_types[tag_list] = tag_type
        else:
            for tag in tag_list:
                tag_types[tag] = tag_type


def gen_data():
    cwd = os.getcwd()
    inputs_dir = os.path.join(cwd, 'e621_data')
    metadata_dir = os.path.join(cwd, 'parsed_data')

    os.makedirs(metadata_dir, exist_ok=True)

    mapping = generate_tag_mapping(get_latest_tags_file(inputs_dir))
    print(f'Generated {len(mapping)} tag mappings')

    implications = generate_tag_implication_tree(get_latest_implications(inputs_dir))
    print(f'Generated {len(implications)} tag implications')

    species_implications_graph = generate_tag_implication_graph(get_latest_implications(inputs_dir), mapping)
    print(f'Generated {len(species_implications_graph)} implications graph')

    # print(get_processed_chain(species_implications_graph, 'husky'))

    posts_file = get_latest_post_file(inputs_dir)
    print(f'Using posts file: {posts_file}')

    skipped_count = {
        'score': 0,
        'ext': 0,
        'nsfw_cub_young': 0,
        'tag': 0,
        'aspect': 0,
        'good': 0,
        'bad_good_tags': 0
    }
    total_count = 0

    good_shit_list = []
    good_shit_ids = set()

    h_nsfw = SortedList(150000)
    h_sfw = SortedList(100000)
    for i, post in tqdm(enumerate(get_posts(posts_file))):
        post: PostData
        total_count += 1

        has_bad_tags = False
        # monochrome black_and_white
        has_bad_good_tags = 'monochrome' in post.tags and 'black_and_white' in post.tags
        for tag in post.tags:
            if tag in bad_tags:
                has_bad_tags = True

        if has_bad_tags:
            skipped_count['tag'] += 1
            continue

        if post.file_ext not in allowed_extensions:
            skipped_count['ext'] += 1
            continue

        if ('cub' in post.tags or 'young' in post.tags) and post.rating != PostRating.SAFE:
            skipped_count['nsfw_cub_young'] += 1
            continue

        if max(post.width, post.height) / min(post.width, post.height) > 2:
            skipped_count['aspect'] += 1
            continue

        if is_good_shit(post):
            if has_bad_good_tags:
                skipped_count['bad_good_tags'] += 1
                continue
            good_shit_list.append(post)
            good_shit_ids.add(post.id)
            skipped_count['good'] += 1
            continue

        if post.score < 0:
            skipped_count['score'] += 1
            continue

        if post.rating == PostRating.EXPLICIT:
            h_nsfw.add(post)
        else:
            h_sfw.add(post)
        # if i % 10000 == 0:
        #     print(i, post.get_formatted_tags(mapping, implications))

    print(f'Total posts in heap: (sfw: {len(h_sfw)}, nsfw: {len(h_nsfw)})')

    tag_types = {}

    total_written = 0
    with open(os.path.join(metadata_dir, 'meta.jsonl'), 'wt') as f:
        for post in tqdm(list(itertools.chain(good_shit_list, h_sfw.get_list(), h_nsfw.get_list()))):
            tags = post.get_formatted_tags(mapping, species_implications_graph)

            update_seen_tags(tag_types, tags)

            total_written += 1
            f.write(json.dumps({
                'id': post.id,
                'w': post.width,
                'h': post.height,
                'bytes': post.file_size,
                'created_at': post.created_at,
                'score': post.score,
                'fav_count': post.fav_count,
                'good': post.id in good_shit_ids,
                'md5': post.md5,
                'url': post.url,
                'ext': post.file_ext,
                'tags': tags,
            }) + '\n')

    print(f'Total posts processed {total_count}, posts written: {total_written}, posts skipped: {skipped_count}')

    with open(os.path.join(metadata_dir, 'tags.json'), 'wt') as f:
        f.write(json.dumps(tag_types, indent=4))


if __name__ == '__main__':
    gen_data()
