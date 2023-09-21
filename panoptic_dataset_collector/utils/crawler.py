import os
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from dataset_collector.utils.utils import safe_requests, valid_extension


class Crawler:
    def __init__(
        self,
        api_key: str,
        engine_id: str,
        search_key: str,
        commercial_only: bool,
        deep_search: bool,
    ):
        # Checks
        assert api_key is not None and api_key != ""
        assert engine_id is not None and engine_id != ""
        assert search_key is not None and search_key != ""

        # Fix license
        license = "(cc_publicdomain%7Ccc_attribute%7Ccc_sharealike%7Ccc_nonderived)"
        self.license_keyword = "creative commons"
        if not commercial_only:
            license += ".-(cc_noncommercial)"
            self.license_keyword = ""

        # Custom search engine url
        self.url = "https://www.googleapis.com/customsearch/v1"
        self.params = {
            "key": api_key,
            "cx": engine_id,
            "q": search_key,
            "searchType": "image",
            "rights": license,
        }
        self.deep_search = deep_search

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

    # Function to extract image URLs from a page
    def _extract_image_urls(self, page_url: str) -> List[str]:
        img_urls = []
        if not safe_requests(page_url):
            return img_urls
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, "html.parser")

        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url and not img_url.startswith("data:"):
                img_urls.append(urljoin(page_url, img_url))

        return img_urls

    def _deep_crawl(self, base_url: str) -> List[str]:
        # image license

        image_urls = []
        crawled_urls = set()
        to_crawl = [base_url]

        while to_crawl:
            current_url = to_crawl.pop()
            if current_url in crawled_urls:
                continue

            crawled_urls.add(current_url)
            image_urls += self._extract_image_urls(current_url)

            # Extract links to other pages and add them to the crawling queue
            if not safe_requests(current_url):
                continue
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, "html.parser")
            for link in soup.find_all("a", href=True):
                link_url = urljoin(current_url, link["href"])
                if urlparse(link_url).netloc == urlparse(base_url).netloc:
                    to_crawl.append(link_url)

        return image_urls

    def crawl(self, start_id: int) -> List[str]:
        params = self.params.copy()
        params["start"] = start_id
        response = requests.get(self.url, params=params)
        data = response.json()
        items = data.get("items", [])
        image_urls = []

        # Loop over all hits
        for id in range(0, len(items)):
            item = items[id]
            image_urls += [item.get("link", "")]
            page_url = item.get("image", {}).get("contextLink", "")
            if self.deep_search:
                image_urls += self._deep_crawl(page_url)
        return image_urls
