from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.data.make_dataset import MakeDataset
from src.data.metadata import Metadata
from src.data.views import ViewCapture
from src.models.object_detection import ObjectDetection
from utils.setup_env import setup_project_env


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.img_dir = Path(config['img_dir'])  # raw data directory
        self.box_dir = Path(config['box_dir'])
        self.img_path = Path(config['img_path'])
        self.vid_path = Path(config['vid_path'])
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_metadata(self):
        metadata = Metadata(self.config)
        metadata.iter_img_dir()
        metadata.get_img_dim()
        metadata.get_mem_usage()

    def process_data(self):
        process = MakeDataset(self.config)
        # process.resize_img(img, new_shape)
        process.img_to_vid()

    def play_capture(self, path, fps):
        view = ViewCapture(self.config)
        view.play_capture(path, fps)

    def main(self):
        self.logger.info('Starting pipeline...')
        img = cv2.imread(self.img_path)

        # self.get_metadata()
        # self.process_data()
        self.play_capture(path=self.vid_path, fps=10)

        # 'data/processed/output_video.avi')
        object_detection = ObjectDetection(self.config, self.img_path)
        results = object_detection.detect_objects(img)
        object_detection.draw_boxes(img, results)

        self.logger.info('Ending pipeline...')

    def test(self):
        # self.logger.info('TEST...')
        # img = cv2.imread(self.img_path)

        # self.get_metadata()
        # # self.process_data()
        # self.play_capture(path=self.vid_path, fps=10)

        # 'data/processed/output_video.avi')

        object_detection = ObjectDetection(self.config, self.img_path)
        for img in self.img_dir.rglob('*.PNG'):
            results = object_detection.detect_objects(img)
            object_detection.draw_boxes(img, results)
            self.play_capture(path=img, fps=10)
            # print(image_path)


if __name__ == '__main__':
    project_dir, config, setup_logs = setup_project_env()
    pipeline = Pipeline(config)
    pipeline.main()
    # pipeline.test()
