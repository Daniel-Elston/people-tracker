from __future__ import annotations

import cv2

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


def write_capture(cap):
    ret, first_frame = cap.read()
    if not ret:
        return
    frame_height, frame_width = first_frame.shape[:2]
    output_path = config['output_path']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return out
