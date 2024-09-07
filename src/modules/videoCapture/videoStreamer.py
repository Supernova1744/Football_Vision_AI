import os
import cv2
import tqdm
from typing import Generator, Tuple
from functools import cached_property
from . import Frame

class videoStreamer:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.load_video()
    
    def load_video(self) -> None:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"The video path provided was not found: {self.video_path}")
        try:
            self.cap = cv2.VideoCapture(self.video_path)
        except Exception as e:
            raise RuntimeError(f"An error happend while opening your video: {e}")
        self.current_frame_id = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @cached_property
    def video_size(self) -> Tuple:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width)

    @cached_property
    def video_fps(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FPS))
    
    def __call__(self, stop_frame_id=-1) -> Generator[Frame, None, None]:
        total_steps = self.total_frames if stop_frame_id == -1 else stop_frame_id
        bar = tqdm.tqdm(total=total_steps, ascii="░▒█")
        while True:
            ret, frame_image = self.cap.read()
            if self.current_frame_id == stop_frame_id:
                return
            if ret:
                frame = Frame(self.current_frame_id, frame_image)
                self.current_frame_id += 1
                bar.update(1)
                yield frame
            else:
                return
    
    def reset(self):
        self.cap.release()
        self.load_video()
        