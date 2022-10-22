import shutil
from os.path import abspath

from huskydb.store import HuskyStore
from huskydb.models import ImageTag, ImageEntry


def run_test():
    shutil.rmtree(abspath('../data/img'), ignore_errors=True)
    with HuskyStore(abspath('../data')) as ds:
        ds.db.clear()
        ds.db.add_tags([
            ImageTag(tag='dog', type=1),
            ImageTag(tag='cat', type=1),
        ])
        entry_id = ds.db.store_entry(ImageEntry.parse_obj({
            'filename': '14.png',
            'original_width': 1024,
            'original_height': 2048,
            'width': 1024,
            'height': 1024,
            'metadata': {'hewwo': 123}
        }), tags=['dog', 'cat'])
        print(ds.db.tag_map)
        print(ds.db.get_entry(entry_id))
        print(f'in db: {ds.db.get_entry_count()}')

        ds.download_all()


def fill_data():
    with HuskyStore(abspath('../data')) as ds:
        ds.db.clear()
        ds.db.fill_with_mock_data(1000)


if __name__ == '__main__':
    fill_data()
