import os
import warnings

import gradio as gr
import lightning as L
import numpy as np
from lang_sam import SAM_MODELS

from dataset_collector.utils.crawler import Crawler
from dataset_collector.utils.filter import Filter
from dataset_collector.utils.panoptic_annotator import PanopticAnnotator
from dataset_collector.utils.serve_gradio_iterative import ServeGradioIterative
from dataset_collector.utils.utils import make_label_file

warnings.filterwarnings("ignore")


class LitGradio(ServeGradioIterative):

    inputs = [
        gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_l"),
        gr.Textbox(lines=1, label="Text prompt", placeholder="safari in india"),
        gr.Slider(0, 100, label="Number of pages (10 images per page)", value=10),
        gr.Textbox(lines=4, label="Class-labels", placeholder="tiger\nelephant\nrhinoceros"),
        gr.Textbox(lines=1, label="Google API key", placeholder="Alpha numeric key"),
        gr.Textbox(
            lines=1, label="Google custom search engine id", placeholder="Alpha numeric key"
        ),
        gr.Checkbox(value=False, label="Commercial license images"),
        gr.Checkbox(value=False, label="Deep search images"),
        gr.Slider(0, 1, value=0.3, label="Box threshold"),
        gr.Slider(0, 1, value=0.25, label="Text threshold"),
    ]
    outputs = [gr.outputs.Image(type="numpy", label="Output Image")]
    enable_queue = False

    title = "Panoptic dataset collector"

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(
        self,
        sam_type,
        search_key,
        search_pages,
        class_labels,
        api_key,
        engine_id,
        commercial_only,
        deep_search,
        box_threshold,
        text_threshold,
    ) -> np.ndarray:
        if self._model is None:
            self._model = self.build_model(
                sam_type,
                search_key,
                class_labels,
                api_key,
                engine_id,
                commercial_only,
                deep_search,
            )
        crawler, filter, annotator = self._model

        # Google search allows only 10 searches per call
        # Loop by changing the start index
        results_per_page = 10
        start_index = 1
        end_index = search_pages * results_per_page + start_index
        for start_id in range(start_index, end_index, results_per_page):
            image_urls = crawler.crawl(start_id)
            image_paths = filter.filter_and_download_images(image_urls)
            for pth in image_paths:
                labeled_image = annotator.generate_annotation(
                    pth, box_threshold, text_threshold
                )
                yield labeled_image
        annotator.combine_all_annotations()

    def build_model(
        self,
        sam_type,
        search_key,
        class_labels,
        api_key,
        engine_id,
        commercial_only,
        deep_search,
    ):
        download_folder = os.path.join(
            os.getcwd(), "dataset_collector", "datasets", search_key.replace(" ", "_")
        )
        label_file = make_label_file(class_labels.splitlines(), search_key)
        crawler = Crawler(api_key, engine_id, search_key, commercial_only, deep_search)
        filter = Filter(commercial_only, download_folder)
        annotator = PanopticAnnotator(download_folder, label_file, sam_type)
        self.ready = True
        return (crawler, filter, annotator)


app = L.LightningApp(LitGradio())
