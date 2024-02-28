from __future__ import annotations

import logging

import cv2
from ultralytics import YOLO


class ObjectDetection():
    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self):
        model = YOLO('yolov8s.pt')
        return model

    def detect_objects(self, img):
        results = self.model(img)
        return results

    def draw_boxes(self, img, results):
        # Access the boxes, confs, and classes from results
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            # Loop through the boxes and draw them on the image
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = confs[i]
                cls = classes[i]

                # Convert tensor to scalar
                conf = conf.item()
                cls = cls.item()

                # Convert to int/float
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                conf = round(float(conf), 2)
                cls = int(cls)

                # get class names from dict
                current_class = self.CLASS_NAMES_DICT[cls]

                # get central dots
                cx = (x1+x2)//2
                cy = (y1+y2)//2

                # if current_class == 'person':
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{current_class} {conf}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imwrite('data/interim/detected_frame_000001.PNG', img)
        # APPLY LOGGING TO GIVE INFO REGARDING THE SHAPES OF IN/OUT
