1. run `download_meta_files.py` - to downloads the database dump from e621
2. run `gen_data.py` - to generate the meta files
3. run `build_dataset.py` - to build the dataset and upload
    * You need to set 34 env vars for this:
    * S3_KEY_ID=
    * S3_KEY_SECRET=
    * S3_BUCKET_NAME=
    * S3_BACKEND_URL=
    * For now, the upload path is hardcoded as `img/<id>.<ext>`
4. copy data/upload_progress.txt to the remote server as data/available_files.txt
    * This step is optional, but is recommended if you have a partially uploaded dataset or some images failed to upload
5. on the remote server, run `download_dataset.py`

Example of a husky.yml file:

```yaml
# must match the .sqlite file name, in this case my_dataset.sqlite
dataset_name: my_dataset

# where the images will be saved, relative to husky.yml
image_dir: ./img

# relative directory of my_dataset.sqlite
index_dir: .

# if set to true, the downloaded images will be loaded and compared to the expected width/height
extended_validation_when_downloading: true

# configuration for the dataset
data_source:
  # base url for downloading files, the image filename will be added at the end of this
  # for example, a file in the db with name 123.png will be downloaded from https://cdn.some.website/path/img/123.png
  base_url: https://cdn.some.website/path/img/

  # an optional text file with the list of the available files so the db doesn't waste time trying to get 
  # the ones that were not uploaded
  available_files_index: ./available_files.txt
```
