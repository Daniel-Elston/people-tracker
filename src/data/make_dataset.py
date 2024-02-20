from __future__ import annotations

import logging
from pathlib import Path

import cv2


class MakeDataset:
    def __init__(self, config):
        self.config = config
        self.img_dir = Path(config['img_dir'])
        self.box_dir = Path(config['box_dir'])
        self.logger = logging.getLogger(self.__class__.__name__)

    def resize_img(self, img, new_shape):
        """Handled by YOLO"""
        pass

    def img_to_vid(self):
        video_path = 'data/processed/output_video.avi'

        images = [img for img in sorted(self.img_dir.rglob('*.PNG'))]
        frame = cv2.imread(str(images[0]))
        if frame is None:
            self.logger.error(
                "Failed to load the first image. Check the image path and format.")
            return
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(video_path, fourcc, 1, (width, height))

        for image in images:
            img = cv2.imread(str(image))
            if img is not None:
                video.write(img)
            else:
                self.logger.warning(f"Could not load image {image}. Skipping.")

        video.release()
        self.logger.info(f"Video saved to {video_path}")
