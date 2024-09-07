import numpy as np
from src.modules.videoCapture import Frame
from .trackerPipeline import trackerPipeline
from .detectionPipeline import detectionPipeline
from src.models.base.serviceProvider import serviceProvider
from src.models.yolov8.utils import compute_iou, xywh2xyxy

class detectionTrackingPipeline:
    def __init__(self, args, config, providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        self.DetectionPipeline = detectionPipeline(
            path=config.DET_MODEL_PATH,
            conf_thres=config.DET_CONF_THRESHOLD,
            iou_thres=config.DET_IOU_THRESHOLD,
            providers=providers
        )
        self.TrackerPipeline = trackerPipeline(args, config)
    
    def get_player_with_ball(self, ball_detections, online_players):
        iou_matrix = compute_iou(ball_detections, xywh2xyxy(online_players[:, :4]))
        _id = np.argmax(iou_matrix)
        if iou_matrix[_id] > 0.0:
            return online_players[_id, 5]
        return -1
            
    def __call__(self, frame: Frame):
        human_detections, ball_detections = self.DetectionPipeline(frame)
        online_players, online_targets = self.TrackerPipeline(frame, human_detections)
        ball_xy = self.DetectionPipeline.BALL_PATHS[-1]
        ball_xy = (ball_xy[0] - 35, ball_xy[1] - 35, ball_xy[0] + 35, ball_xy[1] + 35)
        player_with_ball = self.get_player_with_ball(np.array(ball_xy), online_players)
        return ball_detections, online_players, online_targets, player_with_ball

    @property
    def BALL_PATHS(self):
        return self.DetectionPipeline.BALL_PATHS
