import cv2
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Frame:
    def __init__(self, frame_id, frame_image) -> None:
        self.frame_id = frame_id
        self.frame_image = frame_image
    
    @cached_property
    def rgb_image(self):
        return cv2.cvtColor(self.frame_image, cv2.COLOR_BGR2RGB).copy()
    
    @cached_property
    def shape(self):
        return self.frame_image.shape
