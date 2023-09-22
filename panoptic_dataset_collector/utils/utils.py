import gc
import os
from typing import List, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from panoptic_dataset_collector.utils.io import read_json, write_json, write_yaml

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def safe_requests(image_url: str, connection_timeout: int = 5) -> bool:
    try:
        requests.get(image_url, timeout=connection_timeout)
        return True
    except requests.Timeout:
        return False


def combine_annotations_in_dir(intermediate_json_dir: str, final_json_filename: str):
    intermediate_json_files = os.listdir(intermediate_json_dir)
    annotations = []
    for json_file in intermediate_json_files:
        json_info = read_json(os.path.join(intermediate_json_dir, json_file))
        annotations.append(json_info)
    write_json(final_json_filename, annotations)


def resize_image_keep_aspect_ratio(image: Image, size: List[int]) -> Image:
    old_width, old_height = image.size
    if old_width > size[0]:
        new_width = size[0]
        new_height = int(size[0] * old_height / old_width)
    else:
        new_height = size[1]
        new_width = int(size[1] * old_width / old_height)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    return resized_image


def free_mem() -> None:
    # Free unused cuda memory before tracing
    # This allows Tensor-RT to use different compression tactics
    gc.collect()
    del gc.garbage[:]
    torch.cuda.empty_cache()


def valid_extension(file_name: str) -> bool:
    extension = os.path.splitext(file_name)[1]
    if extension in VALID_EXTENSIONS:
        return True
    return False


def make_label_file(class_labels: List[str], search_key: str) -> str:
    key = search_key.replace(" ", "_")
    dt = {"label_name": key}
    dt["categories"] = [{"name": lbl} for lbl in class_labels]
    key += ".yaml"
    write_yaml(key, dt)
    return key


def get_iou(
    instance_id_mask: np.ndarray, mask: np.ndarray, current_instance_id: int
) -> Tuple[float, int]:
    current_instance_mask = instance_id_mask[mask == 1]
    intersection = sum(current_instance_mask > 0)
    if intersection == 0:
        return (0.0, 0)

    previous_instance_cnt, previous_instance_ids = np.histogram(
        current_instance_mask, current_instance_id, [0, current_instance_id]
    )
    start_id = 1  # skip instance id 0
    overlapping_instance_id = previous_instance_ids[
        np.argmax(previous_instance_cnt[start_id:]) + start_id
    ]
    area_overlapping_instance = (instance_id_mask == overlapping_instance_id).sum()
    area_current_instance = mask.sum()
    union = (area_overlapping_instance + area_current_instance) - intersection
    iou = intersection / union

    remove_instance_id = current_instance_id
    if area_overlapping_instance < area_current_instance:
        remove_instance_id = overlapping_instance_id
    return (iou, remove_instance_id)
