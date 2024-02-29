from __future__ import annotations

import logging
from pathlib import Path

import cv2


class ViewCapture:
    def __init__(self, config):
        self.config = config
        self.img_path = Path(config['img_path'])
        self.vid_path = Path(config['vid_path'])
        self.logger = logging.getLogger(self.__class__.__name__)

    def play_vid(self, path, fps):
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        if fps == 'default' or fps is None:
            frame_time = int(1000 / cap.get(cv2.CAP_PROP_FPS))
        elif fps > 0:
            frame_time = int(1000 / fps)
        elif fps == 0:
            frame_time = 0
        else:
            self.logger.error(f"Invalid fps value: {fps}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def play_img(self):
        img = cv2.imread(str(self.img_path))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def play_capture(self, path, fps):
        form = str(path.suffix[1:])
        try:
            if form in ['avi', 'mp4']:
                self.play_vid(path, fps)
            elif form == 'PNG':
                self.play_img()
            else:
                self.logger.error(f"Invalid form: {form}")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return
