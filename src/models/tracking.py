from __future__ import annotations

import logging

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.tracking_model = self.load_model()
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self):
        tracking_model = DeepSort(max_age=20)
        return tracking_model

    def track_objects(self, data, frame):
        tracks = self.tracking_model.update_tracks(data, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])

            conf = track.get_det_conf()
            if conf is not None and conf > 0.3:

                if track.get_det_class() == 'person':
                    colour = (0, 255, 0)
                else:
                    colour = (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colour, 1)
                cv2.rectangle(frame, (xmin, ymin - 20),
                              (xmin + 20, ymin), colour, -1)
                cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, str(track.det_class), (xmin + 5 + 25, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
