from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.data.make_dataset import MakeDataset
from src.data.metadata import Metadata
from src.data.views import ViewCapture
from src.models.object_detection import ObjectDetection
from src.models.tracking import Tracker
from utils.my_utils import write_capture
from utils.setup_env import setup_project_env


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.img_dir = Path(config['img_dir'])  # raw data directory
        self.box_dir = Path(config['box_dir'])
        self.img_path = Path(config['img_path'])
        self.capture = Path(config['vid_path'])
        self.detector = self.load_detector()
        self.tracker = self.load_tracker()
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_get_metadata(self):
        metadata = Metadata(self.config)
        metadata.get_metadata()

    def process_data(self):
        process = MakeDataset(self.config)
        process.img_to_vid()

    def play_capture(self, path, fps):
        view = ViewCapture(self.config)
        view.play_capture(path, fps)

    def load_detector(self):
        detector = ObjectDetection(self.config, self.capture)
        return detector

    def load_tracker(self):
        tracker = Tracker(self.config, self.capture)
        return tracker

    def main(self):
        self.logger.info('Starting pipeline...')

        fps = cv2.CAP_PROP_FPS
        self.run_get_metadata()
        self.play_capture(self.capture, fps)

        # Open capture
        cap = cv2.VideoCapture(str(self.capture))
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Initialise writer
        out = write_capture(cap)

        # Read and process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply object detection and tracking
            detections = self.detector.detect_objects(frame)
            data = self.detector.get_boxes(detections)
            self.tracker.track_objects(data, frame)

            # Write the processed frame to the output video
            out.write(frame)

            # Display the frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.logger.info('Ending pipeline...')

    def test(self):
        """Show resultant capture with tracking"""
        fps = cv2.CAP_PROP_FPS
        path = Path('reports/figures/result1.mp4')
        self.play_capture(path, fps)


if __name__ == '__main__':
    project_dir, config, setup_logs = setup_project_env()
    pipeline = Pipeline(config)
    pipeline.main()
    # pipeline.test()
