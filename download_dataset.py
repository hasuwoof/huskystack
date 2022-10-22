import os.path

from huskydb.store import HuskyStore


def download_all(data_path: str):
    print(f'Init husky store at {data_path}')

    with HuskyStore(data_path) as store:
        store.download_all(threads=16, use_tqdm=True)


if __name__ == '__main__':
    download_all(os.path.abspath('./data'))
