from typing import Optional, Dict, List

from pydantic import BaseModel


class HuskyConfig(BaseModel):
    dataset_name: str = 'dataset'
    image_dir: str = './img'
    index_dir: str = '.'
    extended_validation_when_downloading: bool = False
    data_source: Optional[dict] = None


class ImageEntry(BaseModel):
    id: Optional[int]
    filename: str
    original_width: int
    original_height: int
    metadata: Optional[dict]
    width: int
    height: int

    tags: Optional[Dict[int, List[str]]]


class ImageTag(BaseModel):
    id: Optional[int]
    tag: str
    type: int
