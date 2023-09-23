import os
from typing import List

import numpy as np
import torch
from lang_sam import SAM_MODELS, LangSAM
from lang_sam.utils import draw_image

from panoptic_dataset_collector.utils.io import (
    delete_file,
    read_image,
    read_yaml,
    save_ndarray_image,
    write_json,
)
from panoptic_dataset_collector.utils.utils import (
    combine_annotations_in_dir,
    free_mem,
    get_iou,
)


class PanopticAnnotator:
    def __init__(self, download_folder: str, label_file: str, sam_type: str = "vit_l"):

        # Checks
        assert sam_type.lower() in list(SAM_MODELS.keys())
        assert label_file is not None and label_file != ""

        # Intermediate variables
        self.panoptic_annotation_path = os.path.join(download_folder, "panoptic_annotation")
        self.intermediate_json_path = os.path.join(download_folder, "annotation_json")
        self.final_annotation_file_name = os.path.join(
            download_folder, "panoptic_annotation.json"
        )
        self.image_cnt = 0
        self.valid_iou = 0.6

        # Create result folders
        os.makedirs(self.panoptic_annotation_path, exist_ok=False)
        os.makedirs(self.intermediate_json_path, exist_ok=False)

        # Load model
        self.labels = []
        label_info = read_yaml(label_file)
        self.labels = label_info["categories"]
        assert len(self.labels) > 0
        self.model = LangSAM(sam_type=sam_type.lower())

    def model(self) -> LangSAM:
        return self.model

    # Function to generate panoptic results
    def _generate_image_labels(
        self, img_path: str, box_threshold: float, text_threshold: float
    ) -> np.ndarray:
        image_pil = read_image(img_path, rgb=True)
        image_array = np.asarray(image_pil)
        masks = []
        boxes = []
        class_id = []
        labels = []

        # TODO: Avoid this loop
        for id, lbl in enumerate(self.labels):
            free_mem()
            masks_lbl, boxes_lbl, phrases, logits = self.model.predict(
                image_pil, lbl["name"], box_threshold, text_threshold
            )
            labels += [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
            assert masks_lbl.shape[0] == boxes_lbl.shape[0]
            if len(masks) == 0:
                masks = masks_lbl
                boxes = boxes_lbl
            elif masks_lbl.shape[0]:
                masks = torch.cat([masks, masks_lbl])
                boxes = torch.cat([boxes, boxes_lbl])
            class_id += [id] * len(masks_lbl)

        if len(masks):
            valid_ids = self._add_coco_segment(img_path, masks, boxes, class_id)
            image_array = draw_image(
                image_array,
                masks[valid_ids, ...],
                boxes[valid_ids, :],
                [labels[id] for id in valid_ids],
            )
            return image_array

        # No valid labels in the downloaded image
        # Delete file
        delete_file(img_path)
        return image_array

    # Function to add panoptic annotation in coco format
    def _add_coco_segment(
        self, img_path: str, masks: torch.Tensor, boxes: torch.Tensor, class_id: List[int]
    ) -> List[int]:
        assert masks.shape[0] == boxes.shape[0] == len(class_id)
        self.image_cnt += 1
        img_name = img_path.split("/")[-1]
        annotation = dict(
            image_id=self.image_cnt,
            file_name=img_name,
            segments_info=[],
        )
        panoptic_image = np.zeros((masks.shape[1], masks.shape[2])).astype(np.uint8)
        segment_info = dict()
        valid_ids = dict()
        for id in range(len(class_id)):
            new_instance_id = id + 1
            new_instance_mask = masks[id].numpy()
            iou, replace_instance = get_iou(panoptic_image, new_instance_mask, new_instance_id)
            # Add new instance only if no significant overlapping previous instance
            if iou < self.valid_iou or replace_instance != new_instance_id:
                bbox = [int(id) for id in boxes[id, :].numpy()]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                segment_info[new_instance_id] = dict(
                    id=new_instance_id,
                    category_id=class_id[id],
                    bbox=bbox,
                    iscrowd=0,
                    area=area,
                )
                panoptic_image[new_instance_mask == 1] = new_instance_id
                valid_ids[new_instance_id] = id

            # Remove previous overlapping instance
            if iou > self.valid_iou and replace_instance != new_instance_id:
                segment_info.pop(replace_instance)
                valid_ids.pop(replace_instance)
                panoptic_image[panoptic_image == replace_instance] = 0

        save_ndarray_image(
            os.path.join(self.panoptic_annotation_path, img_name), panoptic_image
        )
        annotation["segments_info"] = list(segment_info.values())
        json_filename = img_name.split(".")[0] + ".json"
        json_filename = os.path.join(self.intermediate_json_path, json_filename)
        write_json(json_filename, annotation)
        return list(valid_ids.values())

    def generate_annotation(
        self, image_path: str, box_threshold: float, text_threshold: float
    ) -> np.ndarray:
        labeled_image = self._generate_image_labels(image_path, box_threshold, text_threshold)
        return labeled_image

    # Function to combine all intermediate jsons
    def combine_all_annotations(self):
        combine_annotations_in_dir(
            self.intermediate_json_path, self.final_annotation_file_name
        )
