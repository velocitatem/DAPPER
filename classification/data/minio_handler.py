import io
import re
import logging
from minio import Minio
from PIL import Image
from typing import Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class MinioManager:
    def __init__(self,
                 endpoint = "localhost:9900",
                 access_key = "minioadmin",
                 secret_key= "minioadmin",
                 bucket_name = "dapper",
                 secure: bool = False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self._refresh_object_list()

    def _refresh_object_list(self):
        self.all_images = set(obj.object_name for obj in self.client.list_objects(self.bucket_name))

    def _generate_filename(self, label: str, index: int, source: str, is_augmented: bool) -> str:
        clean_label = re.sub(r'\\W+', '', str(label))
        aug_info = "aug" if is_augmented else "orig"
        return f"src_{source}_cls_{clean_label}_idx_{index}_{aug_info}.jpg"

    def upload_image(self, image: Image.Image, filename: str) -> str:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.client.put_object(
            self.bucket_name,
            filename,
            image_bytes,
            len(image_bytes.getvalue())
        )
        self.all_images.add(filename)
        return f"http://{self.endpoint}/{self.bucket_name}/{filename}"

    def process_row(self, index: int, row: pd.Series) -> Optional[tuple]:
        image = row['image']
        if image is None or not isinstance(image, Image.Image):
            logging.warning(f"Row {index} has no valid image, skipping.")
            return index, None

        label = str(row['label'])
        source = row.get('source_dataset', 'unknown')
        is_augmented = row.get('is_augmented', False)

        filename = self._generate_filename(label, index, source, is_augmented)
        if filename in self.all_images:
            return index, f"http://{self.endpoint}/{self.bucket_name}/{filename}"

        image = image.convert("RGB")
        return index, self.upload_image(image, filename)

    def upload_dataframe(self, df: pd.DataFrame, max_workers: int = 12) -> pd.DataFrame:
        args = [(i, row) for i, row in df.iterrows()]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(lambda args: self.process_row(*args), args)
        for i, url in results:
            if url is not None:
                df.at[i, 'image'] = url
        return df
