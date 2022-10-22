from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict

import networkx as nx

from huskygen.constants import tag_remap, excluded_tags, special_tags, number_re, stripped_parts, excluded_species_tags
from huskygen.graph_utils import get_processed_chain
from huskygen.utils import format_human_readable


class PostRating(Enum):
    SAFE = 's'
    QUESTIONABLE = 'q'
    EXPLICIT = 'e'


class TagCategory(Enum):
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


@dataclass
class TagMeta:
    id: int
    name: str
    category: TagCategory
    post_count: int


@dataclass
class PostData:
    id: int
    created_at: str
    md5: str
    url: List[str]
    rating: PostRating
    width: int
    height: int
    tags: List[str]
    fav_count: int
    score: int
    comment_count: int
    file_ext: str
    file_size: int
    duration: Optional[float]

    def __post_init__(self):
        if not isinstance(self.id, int):
            self.id = int(self.id)

    def get_formatted_tags(self, tag_map: Dict[str, TagMeta], species_graph: nx.DiGraph):
        rating = 'explicit' if self.rating == PostRating.EXPLICIT else 'questionable' if self.rating == PostRating.QUESTIONABLE else 'safe'
        artists = []
        species = set()
        special = []
        general = []
        leftover = set()
        all_species = set()

        for tag in self.tags:
            if tag in tag_remap:
                tag = tag_remap[tag]

            if tag.endswith('_humanoid'):
                continue

            if tag in excluded_tags:
                continue

            if tag in excluded_species_tags:
                leftover.add(tag)
                continue

            if tag in special_tags:
                special.append(format_human_readable(tag))
                continue

            if tag.strip() == '':
                continue

            if number_re.match(tag) is not None:
                continue

            if tag.endswith('(disambiguation)'):
                continue

            if tag.startswith('generation') and tag.endswith('_pokemon'):
                continue

            try:
                tag_meta = tag_map[tag]
            except KeyError:
                print(f'DEBUG: Tag not found in map: {tag}')
                leftover.add(tag)
                # filtered_tags.append(format_human_readable(tag))
                continue

            # if this tag is used less than 25 times, drop it.
            # my reasoning behind this is: we don't want to overload the tokenizer, and the network requires A LOT of
            # images to start understanding a concept
            if tag_meta.post_count < 25:
                continue

            if tag_meta.category == TagCategory.INVALID:
                continue

            for part in stripped_parts:
                if tag.endswith(part):
                    tag = re.sub(r'_$', '', tag[:tag.index(part)]).strip()

            if tag_meta.category == TagCategory.ARTIST:
                artists.append(format_human_readable(tag))
                continue

            if tag_meta.category == TagCategory.SPECIES:
                # unconditionally add the species to all species list for later set fuckery
                all_species.add(format_human_readable(tag))

                chain = [x for x in get_processed_chain(species_graph, tag) if
                         x in tag_map and tag_map[x].post_count > 100]

                # if we have a chain of species longer like 2,
                # ex [australian_shepherd, herding_dog, dog, canine], only keep the ends
                if len(chain) > 2:
                    first, *mid, last = chain

                    # add first and last element
                    species.add(format_human_readable(first))
                    species.add(format_human_readable(last))

                    # put everything else into leftover to be randomly sprinkled into the general tags
                    leftover |= set([format_human_readable(x) for x in mid])
                else:
                    # if we have a chain of length 1 or 2, just add the whole chain
                    species |= set([format_human_readable(x) for x in chain])
                continue

            general.append((tag_meta.post_count, format_human_readable(tag)))

        general = [x[1] for x in sorted(general, key=lambda e: e[0], reverse=True)]
        remaining_species = species.difference(leftover)
        leftover = leftover.union(all_species.difference(remaining_species))
        # print(f'DEBUG: {list(all_species)} -> {remaining_species} [leftover tags: {leftover}]')

        return {
            "rating": rating,
            "artists": artists,
            "species": list(remaining_species),
            "special": special,
            "general": general,
            "leftover": list(leftover),
        }

    def __lt__(self, other: PostData):
        return (self.fav_count, self.score, self.id) < (other.fav_count, other.score, other.id)

    def __repr__(self):
        return str((self.fav_count, self.score, self.id))

    def __hash__(self):
        return hash(self.id)
