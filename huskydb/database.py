import copy
import json
import random
import re
import sqlite3
from os import path
from typing import Optional, List, Dict, Generator, Tuple

import requests
import tqdm

from huskydb.models import ImageEntry, ImageTag


class HuskyDatabase:
    __sql_insert_tag = 'insert into tags (tag, type) values (:tag, :type)'

    __sql_insert_tag_entry = 'insert into entries_tags (entry_id, tag_id) values (:entry_id, :tag_id)'

    __sql_insert_entry = """
    insert into entries (filename, original_width, original_height, metadata, width, height)
    values (:filename, :original_width, :original_height, :metadata, :width, :height);
    """

    def __init__(self, index_file: str, disable_foreign_key=False):
        super().__init__()

        if not self.is_thread_safe():
            raise RuntimeError('DB is thread unsafe, this is not supported')

        self.disable_foreign_key = disable_foreign_key

        self.index_file: str = index_file

        self.db: Optional[sqlite3.Connection] = None
        self.tag_map: Dict[str, ImageTag] = dict()

    def init(self):
        self.load_index()

    def _soft_init(self):
        self.db = sqlite3.connect(self.index_file, check_same_thread=False)

    def clear(self):
        self.db.execute('delete from entries_tags')
        self.db.execute('delete from entries')
        self.db.execute('delete from tags')
        self.db.execute('delete from sqlite_sequence where name="tags"')
        self.db.execute('delete from sqlite_sequence where name="entries"')
        self.db.commit()
        self.tag_map = dict()

    def add_tags(self, tags: List[ImageTag]):
        new_tag_map = copy.copy(self.tag_map)
        cur = self.db.cursor()

        try:
            for tag in tags:
                if tag.tag in new_tag_map:
                    raise ValueError(f'Duplicated tag: {tag.tag}')

                cur = cur.execute(self.__sql_insert_tag, tag.dict())
                new_tag_map[tag.tag] = tag.copy(update={'id': cur.lastrowid})
            self.db.commit()
            self.tag_map = new_tag_map
        except Exception:
            self.db.rollback()
            raise

    def commit(self):
        self.db.commit()

    def get_cursor(self):
        return self.db.cursor()

    def store_entry(self, entry: ImageEntry, tags: List[str], cursor: Optional[sqlite3.Cursor] = None) -> int:
        cur = cursor or self.db.cursor()
        try:
            d = entry.dict()

            if 'metadata' in d:
                d['metadata'] = json.dumps(d['metadata'])

            cur.execute(self.__sql_insert_entry, d)
            new_entry_id = cur.lastrowid
            for tag in tags:
                tag_meta = self.tag_map[tag]
                cur.execute(self.__sql_insert_tag_entry, {
                    'entry_id': new_entry_id,
                    'tag_id': tag_meta.id
                })

            if not cursor:
                self.db.commit()
        except Exception:
            if not cursor:
                self.db.rollback()
            raise

        return new_entry_id

    def get_entry(self, entry_id) -> ImageEntry:
        row = self.db.execute('select id, filename, original_width, original_height, metadata, width, height '
                              'from entries where id=?', [entry_id]).fetchone()
        row_id, filename, original_width, original_height, metadata, width, height = row
        entry_dict = {
            'id': row_id,
            'filename': filename,
            'original_width': original_width,
            'original_height': original_height,
            'width': width,
            'height': height,
        }

        if metadata is not None:
            entry_dict['metadata'] = json.loads(metadata)

        tag_dict = self.get_tags(entry_id)

        entry_dict['tags'] = tag_dict

        return ImageEntry.parse_obj(entry_dict)

    def get_entry_no_extra(self, entry_id) -> ImageEntry:
        row = self.db.execute('select id, filename, original_width, original_height, metadata, width, height '
                              'from entries where id=?', [entry_id]).fetchone()
        row_id, filename, original_width, original_height, metadata, width, height = row
        entry_dict = {'id': row_id, 'filename': filename, 'original_width': original_width,
                      'original_height': original_height, 'width': width, 'height': height, 'tags': dict()}

        return ImageEntry.parse_obj(entry_dict)

    def get_tags(self, entry_id: int):
        tag_dict = dict()
        cur = self.db.execute('select entry_id, tag from entries_tags '
                              'inner join tags on tags.id = entries_tags.tag_id where entry_id=?', [entry_id])
        for entry_id, tag_name in cur:
            tag = self.tag_map[tag_name]
            if tag.type not in tag_dict:
                tag_dict[tag.type] = []

            tag_dict[tag.type].append(tag.tag)
        return tag_dict

    def get_entry_count(self):
        return self.db.execute('select count(*) from entries').fetchone()[0]

    def get_filenames(self) -> Generator[Tuple[int, str], None, None]:
        cur = self.db.execute('select id, filename from entries')
        yield from cur

    def load_index(self):
        if path.isfile(self.index_file):
            self.load_index_from_db()
        else:
            self.create_new_dataset()

    def load_index_from_db(self):
        self.db = sqlite3.connect(self.index_file, check_same_thread=False)

        if not self.disable_foreign_key:
            self.db.execute('pragma foreign_keys = 1;')

        cur = self.db.execute('select * from tags')

        for row_id, tag_name, tag_type in cur:
            self.tag_map[tag_name] = ImageTag(id=row_id, tag=tag_name, type=tag_type)

    def create_new_dataset(self):
        self.db = sqlite3.connect(self.index_file, check_same_thread=False)

        if not self.disable_foreign_key:
            self.db.execute('pragma foreign_keys = 1;')

        schema_file = path.abspath(path.join(path.dirname(path.realpath(__file__)), 'schema.sql'))
        with open(schema_file, encoding='utf-8') as f:
            init_sql = f.read()
        self.db.executescript(init_sql)

    def close(self):
        self.db.commit()
        self.db.close()

    def fill_with_mock_data(self, count: int = 10000):
        words = requests.get("https://www.mit.edu/~ecprice/wordlist.10000").content.decode().splitlines()
        regex = re.compile(r"^\w+$")
        words_clean = [w for w in words if regex.match(w) is not None]

        random.shuffle(words_clean)
        tags = list(words_clean[:count])

        self.add_tags([
            ImageTag(tag=tag, type=random.randint(0, 5)) for tag in tags
        ])

        for i in tqdm.tqdm(range(count)):
            w = random.randint(300, 1000)
            h = random.randint(300, 1000)

            self.store_entry(ImageEntry.parse_obj({
                'filename': f'{i}.png',
                'original_width': w,
                'original_height': h,
                'width': w,
                'height': h,
                'metadata': {'hewwo': 123}
            }), tags=list({random.choice(tags) for _ in range(random.randint(5, 10))}))

    def get_entries_iterator(self):
        cur = self.db.execute('select id from entries')

        for i, in cur:
            yield self.get_entry(i)

    def __getstate__(self):
        return {**self.__dict__, 'db': None}

    def __setstate__(self, state):
        self.__dict__ = state
        self._soft_init()

    @classmethod
    def get_sqlite3_thread_safety(cls):
        # from https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
        sqlite_threadsafe2python_dbapi = {0: 0, 2: 1, 1: 3}
        conn = sqlite3.connect(":memory:")
        threadsafety = conn.execute(
            """
            select * from pragma_compile_options
            where compile_options like 'THREADSAFE=%'
            """
        ).fetchone()[0]
        conn.close()

        threadsafety_value = int(threadsafety.split("=")[1])

        return sqlite_threadsafe2python_dbapi[threadsafety_value]

    @classmethod
    def is_thread_safe(cls):
        return cls.get_sqlite3_thread_safety() == 3
