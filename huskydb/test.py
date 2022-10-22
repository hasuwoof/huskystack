import unittest
from os.path import abspath

from huskydb.store import HuskyStore
from huskydb.models import ImageTag, ImageEntry


class HuskyDataSetTest(unittest.TestCase):
    def setUp(self):
        self.hd = HuskyStore(abspath('./data'))
        self.hd.init()
        self.hd.db.clear()

    def tearDown(self):
        self.hd.close()

    def test_db_insert_tag(self):
        self.hd.db.add_tags([
            ImageTag(tag='dog', type=1),
            ImageTag(tag='cat', type=1),
        ])

        self.assertEqual(len(self.hd.db.tag_map.keys()), 2)
        self.assertEqual(self.hd.db.db.execute('SELECT COUNT(*) FROM tags').fetchone()[0], 2)

    def test_db_insert_entry(self):
        self.hd.db.add_tags([
            ImageTag(tag='dog', type=1),
            ImageTag(tag='cat', type=1),
        ])

        entry_id = self.hd.db.store_entry(ImageEntry.parse_obj({
            'filename': 'woof.png',
            'original_width': 1024,
            'original_height': 2048,
            'width': 1024,
            'height': 1024,
            'metadata': {'hewwo': 123}
        }), tags=['dog'])

        self.assertIsNotNone(entry_id)
        self.assertEqual(self.hd.db.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0], 1)
        self.assertEqual(self.hd.db.db.execute('SELECT COUNT(*) FROM entries_tags').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
