from __future__ import annotations

import logging
from pathlib import Path

import cv2


class Metadata:
    def __init__(self, config):
        self.config = config
        self.img_dir = Path(config['img_dir'])
        self.box_dir = Path(config['box_dir'])
        self.logger = logging.getLogger(self.__class__.__name__)

    def iter_img_dir(self):
        dir_1_count = sum(1 for _ in self.img_dir.rglob('*.PNG'))
        dir_2_count = sum(1 for _ in self.box_dir.rglob('*.PNG'))
        self.logger.debug(
            f"Images found: {dir_1_count}, Boxes found: {dir_2_count}")

    def get_img_dim(self):
        incorrect_shape_count = 0
        correct_shape = (720, 1280, 3)

        for image_path in self.img_dir.rglob('*.PNG'):
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(
                        f"Image {image_path} could not be loaded.")
                assert img.shape == correct_shape, f"Image: {image_path} has unexpected shape: {img.shape}"
            except AssertionError as e:
                self.logger.error(e)
                incorrect_shape_count += 1
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                incorrect_shape_count += 1

        if incorrect_shape_count > 0:
            self.logger.debug(
                f'N images with wrong shape or failed to load: {incorrect_shape_count}')
        else:
            self.logger.debug(f'All images have correct shape {correct_shape}')

    def get_mem_usage(self):
        images = [img for img in sorted(self.img_dir.rglob('*.PNG'))]
        mem_usage = 0
        for image in images:
            img = cv2.imread(str(image))
            mem_usage += img.size
        mem_usage = mem_usage / 1024 / 1024
        mem_usage_img = mem_usage / len(images)

        self.logger.debug(f"Total memory usage: {mem_usage}MB")
        self.logger.debug(f"Total memory usage per image: {mem_usage_img}MB")
