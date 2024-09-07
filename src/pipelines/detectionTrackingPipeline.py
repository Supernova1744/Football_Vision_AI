from src.modules.videoCapture import Frame
from .trackerPipeline import trackerPipeline
from .detectionPipeline import detectionPipeline
from src.models.base.serviceProvider import serviceProvider

class detectionTrackingPipeline:
    def __init__(self, args, config, providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        self.DetectionPipeline = detectionPipeline(
            path=config.DET_MODEL_PATH,
            conf_thres=config.DET_CONF_THRESHOLD,
            iou_thres=config.DET_IOU_THRESHOLD,
            providers=providers
        )
        self.TrackerPipeline = trackerPipeline(args, config)
    
    def __call__(self, frame: Frame):
        human_detections, ball_detections = self.DetectionPipeline(frame)
        online_players, online_targets = self.TrackerPipeline(frame, human_detections)
        return ball_detections, online_players, online_targets

    @property
    def BALL_PATHS(self):
        return self.DetectionPipeline.BALL_PATHS
