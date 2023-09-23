import argparse
import os

from panoptic_dataset_collector.utils.crawler import Crawler
from panoptic_dataset_collector.utils.filter import Filter
from panoptic_dataset_collector.utils.panoptic_annotator import PanopticAnnotator


# Main function
def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Generate a database of annotated images with COCO panoptic format."
    )
    parser.add_argument(
        "--search",
        help="Search term for image web search",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--search_pages", help="Number of pages to search (1-10)", default=10, type=int
    )
    parser.add_argument(
        "--label_file",
        help="File with a list of labels needed in annotated images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--api_key",
        help="Your Google API key",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--engine_id",
        help="Your Google custom search engine id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--commercial_only",
        help="Download images with only commercial license",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--deep_search",
        help="Use the returned page urls to search for more images",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--sam_type",
        help="Type of SAM to use [vit_h, vit_l, vit_b]. vit_h is GPU memory intensive but accurate.",
        default="vit_l",
        type=str,
    )
    parser.add_argument(
        "--box_threshold",
        help="Threshold for bounding box.",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--text_threshold",
        help="Threshold for text.",
        default=0.25,
        type=float,
    )
    args = parser.parse_args()

    download_folder = os.path.join(
        os.getcwd(), "panoptic_dataset_collector", "datasets", args.search.replace(" ", "_")
    )
    crawler = Crawler(
        args.api_key, args.engine_id, args.search, args.commercial_only, args.deep_search
    )
    filter = Filter(args.commercial_only, download_folder)
    annotator = PanopticAnnotator(download_folder, args.label_file, args.sam_type)
    # Google search allows only 10 searches per call
    # Loop by changing the start index
    results_per_page = 10
    start_index = 1
    end_index = args.search_pages * results_per_page + start_index
    for start_id in range(start_index, end_index, results_per_page):
        image_urls = crawler.crawl(start_id)
        image_paths = filter.filter_and_download_images(image_urls)
        for pth in image_paths:
            annotator.generate_annotation(pth, args.box_threshold, args.text_threshold)
    annotator.combine_all_annotations()


if __name__ == "__main__":
    main()
