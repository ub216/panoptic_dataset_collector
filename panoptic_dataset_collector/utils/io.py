import json
import os
from typing import Any, Dict, List

import numpy as np
import yaml
from PIL import Image


def read_yaml(yaml_path: str) -> Dict:
    if not os.path.isfile(yaml_path):
        raise RuntimeError(f"Label file does not exist: {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(yaml_path: str, info: Dict):
    with open(yaml_path, "w") as file:
        yaml.dump(info, file)


def read_json(json_path: str) -> List[Dict]:
    if not os.path.isfile(json_path):
        raise RuntimeError(f"Label file does not exist: {json_path}")
    with open(json_path, "r") as fl:
        return json.load(fl)


def write_json(json_path: str, annotation: List[Dict]):
    with open(json_path, "w") as f:
        json.dump(annotation, f)


def read_image(image_path: str, rgb: bool = False) -> Image:
    if not os.path.isfile(image_path):
        raise RuntimeError(f"Image file does not exist: {image_path}")
    image = Image.open(image_path)
    if rgb:
        image = image.convert("RGB")
    return image


def save_image(image_path: str, image: np.ndarray) -> None:
    try:
        image.save(image_path)
    except:
        print(f"Unable to save image to {image_path}")


def save_ndarray_image(image_path: str, image: np.ndarray) -> None:
    image_pil = Image.fromarray(image)
    save_image(image_path, image_pil)


def delete_file(file_path: str):
    os.remove(file_path)
