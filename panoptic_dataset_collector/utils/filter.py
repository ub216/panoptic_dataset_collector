import os
from typing import List

import requests

from dataset_collector.utils.io import delete_file, read_image, save_image
from dataset_collector.utils.utils import (
    resize_image_keep_aspect_ratio,
    safe_requests,
    valid_extension,
)


class Filter:
    def __init__(
        self,
        commercial_only: bool,
        download_folder: str,
    ):
        # Checks
        assert download_folder is not None and download_folder != ""

        # Fix license
        self.license_keyword = ""
        if commercial_only:
            self.license_keyword = "creative commons"

        # Intermediate variables
        self.images_path = os.path.join(download_folder, "images")
        os.makedirs(self.images_path, exist_ok=False)
        self.min_size = [200, 200]
        self.max_size = [1333, 1333]

    # Function to save the image urls
    def _download_image_from_url(self, image_url: str) -> str:
        image_path = ""
        if not safe_requests(image_url):
            return image_path
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            image_filename = image_url.split("/")[-1].split("?")[0]
            if valid_extension(image_filename):
                image_path = os.path.join(self.images_path, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_data)
        return image_path

    # Function to filter images based on license
    def _filter_images_by_license(self, img_urls: List[str]) -> List[str]:
        filtered_urls = []
        for img_url in img_urls:
            if not safe_requests(img_url):
                continue
            response = requests.head(img_url)
            if response.headers.get("License", "").lower() in [self.license_keyword, ""]:
                filtered_urls.append(img_url)
        return filtered_urls

    # Function to filter images based on size
    def _filter_image_by_size(self, img_path: str) -> bool:
        try:
            image = read_image(img_path)
            width, height = image.size
            if width > self.max_size[0] or height > self.max_size[1]:
                image = resize_image_keep_aspect_ratio(image, self.max_size)
                save_image(img_path, image)
                return True
            if width > self.min_size[0] and height > self.min_size[1]:
                return True
        except:
            pass
        # Downloaded image file not usable
        # Delete file
        delete_file(img_path)
        return False

    def filter_and_download_images(self, image_urls: List[str]) -> List[str]:
        filtered_urls = self._filter_images_by_license(image_urls)
        # Download valid images
        image_paths = []
        for img_url in filtered_urls:
            img_path = self._download_image_from_url(img_url)
            if img_path != "" and self._filter_image_by_size(img_path):
                image_paths += [img_path]
                print(f"Downloaded image {img_url}")
        return image_paths
