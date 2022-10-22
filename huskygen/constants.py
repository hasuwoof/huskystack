from __future__ import annotations

import re
from enum import Enum

# Removes numbers or years
number_re = re.compile(r'^\d+(:\d+)?$')

# If the tags starts with a number, inflect it. 1_dog will become one_dog
inflectable_tag_re = re.compile(r'^\d+_')

# Tags which should be dropped from the post. Having one of these tags on the post doesn't mean that it should be dropped.
excluded_tags = {'<3', 'absurd_res', 'year', 'text', 'dialogue', 'hi_res', 'pokemon_(species)', 'conditional_dnp',
                 'unknown_artist', 'unknown_species', 'english_text'}
excluded_species_tags = {'canid', 'canis', 'mammal', 'felid', 'felis', 'equid'}

# XXX: Should be covered by the stripped_parts list
tag_remap = {
    'digital_media_(artwork)': 'digital_media'
}

# Tags that should be pulled out to the very front, no matter of their location
special_tags = {
    'feral', 'anthro'
}

# Parts that should be stripped after the tags.
# TODO: Maybe make this an inclusion list instead?
stripped_parts = [
    '(artwork)',
    '(artist)',
    '(character)',
    '(species)',
    '(general)',
    '(copy)',
]

# Only download these tags
allowed_extensions = {'png', 'jpeg', 'jpg'}

# Any post with these tags will be removed from the dataset
bad_tags = {'feces', 'comic', 'nazi', 'hyper', 'compression_artifacts', 'garfield'}

# Any posts with the following tags will be added to the dataset after the baseline filtration based on favcount
good_tags = {
    # artists
    'angiewolf', 'falvie', 'chunie', 'bloominglynx', 'rakisha',
    'tokifuji', 'trigaroo', 'zackary911', 'demicoeur', 'zoe_(nnecgrau)', 'feralise',
    'peritian', 'rajii', 'xeshaire', 'bassenji', 'alibi-cami', 'kluclew',
    'sigma_x', 'clockhands', 'kazarart', 'codyblue-731', 'fluff-kevlar', 'accelo', 'wolfy-nail', 'suelix',
    'valkoinen', 'miramint',

    # community rec
    'photonoko', 'killioma', 'dimwitdog', 'elvche', 'thousandfoldfeathers', 'catcouch', 'brolaren', 'necrodrone',
    'JadeDragoness', 'ammylin', 'aky', 'wingsandfire72', 'mcfan', 'drakawa', 'blitzdrachin', 'etheross', 'nox',
    'braeburned', 'merrunz',
    'wildering', 'honovy', '100racs', 'dimikendal101', 'spectrumshift', 'lynncore', 'bassenji', 'lynncore', 'ammylin',
    'turdusphilomelos', 'shermugi',
    'lynncore', 'karukuji', 'alectorfencer', 'narse', 'madnessdemon', 'jace', 'zyraxus', 'themefinland', 'thorphax',
    'fossa666', 'underavenwood',

    # characters
    'steele_(accelo)',

    # species
    'nargacuga'
}


class TagType(Enum):
    RATING = 0
    ARTIST = 1
    SPECIES = 2
    SPECIAL = 3
    GENERAL = 4
    LEFTOVER = 5


tag_key_to_type = {
    "rating": TagType.RATING.value,
    "artists": TagType.ARTIST.value,
    "species": TagType.SPECIES.value,
    "special": TagType.SPECIAL.value,
    "general": TagType.GENERAL.value,
    "leftover": TagType.LEFTOVER.value
}

type_to_tag_key = {
    v: k for k, v in tag_key_to_type.items()
}
