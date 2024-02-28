from __future__ import annotations

import logging

import cv2

from src.models.object_detection import ObjectDetection


class SimpleTracker:
    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.detector = self.load_detector()
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_detector(self):
        detector = ObjectDetection(self.config, self.capture)
        return detector

    def tracker(self):

        # Open video capture
        cap = cv2.VideoCapture(str(self.capture))
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # Read and process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply object detection on the frame
            results = self.detector.detect_objects(frame)
            self.detector.draw_boxes(frame, results)

            # Display the frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
