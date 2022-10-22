from huskygen.download_dataset import ensure_directory_exists, remove_old_files, fetch_dataset

if __name__ == '__main__':
    ensure_directory_exists()
    remove_old_files()

    fetch_dataset()
