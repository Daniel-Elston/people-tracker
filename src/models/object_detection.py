from __future__ import annotations

import logging

from ultralytics import YOLO


class ObjectDetection():
    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.detection_model = self.load_model()
        self.CLASS_NAMES_DICT = self.detection_model.model.names
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self):
        detection_model = YOLO('yolov8n.pt')
        return detection_model

    def detect_objects(self, frame):
        detections = self.detection_model(frame)[0]
        return detections

    def get_boxes(self, detections):

        results = []
        # access boxes, confs, and classes
        for data in detections.boxes.data.tolist():
            xmin, ymin, xmax, ymax = int(data[0]), int(
                data[1]), int(data[2]), int(data[3])
            conf = float(data[4])
            cls = self.CLASS_NAMES_DICT[int(data[5])]
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], conf, cls])

        return results
