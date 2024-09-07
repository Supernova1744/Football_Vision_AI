import numpy as np
from typing import List
from src.modules.videoCapture import Frame
from src.models.base.serviceProvider import serviceProvider
from src.utils import *
from src.models.yolov8.utils import xywh2xyxy
from src.models import YOLOv8

class detectionPipeline:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        self.BALL_PATHS = []
        self.BALL_CLS_ID = 0
        self.model = YOLOv8(
            path=path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            providers=providers
        )

    def handleBallDetections(self, ball_detections: np.array) -> np.array:
        if ball_detections.shape[0]:
            ball_detections = ball_detections[np.argmax(ball_detections[:, 4])][np.newaxis, :]
            self.BALL_PATHS.append(ball_detections[0, :2])
        else:
            self.BALL_PATHS.append(calc_direction(self.BALL_PATHS)) # Add new points as current estimated ball location
        return ball_detections

    def __call__(self, frame: Frame) -> List:
        # Object Detection
        boxes, scores, class_ids = self.model(frame.rgb_image)
        xyxy = xywh2xyxy(boxes)

        # Prepare For Tracking
        human_ids = np.where(class_ids != self.BALL_CLS_ID)
        ball_ids  = np.where(class_ids == self.BALL_CLS_ID)

        # Handle Ball Detection
        ball_detections = np.concatenate([boxes, scores[:, np.newaxis], class_ids[:, np.newaxis]], axis=-1)[ball_ids]
        ball_detections = self.handleBallDetections(ball_detections)
        # Concatenate data
        human_detections = np.concatenate([xyxy, scores[:, np.newaxis], class_ids[:, np.newaxis]], axis=-1)[human_ids]
        return human_detections, ball_detections