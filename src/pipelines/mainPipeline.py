import numpy as np
from typing import List
from src.modules.videoCapture import Frame
from src.models.base.serviceProvider import serviceProvider
from .posePipeline import posePipeline
from .detectionTrackingPipeline import detectionTrackingPipeline

from concurrent.futures import ThreadPoolExecutor

class mainPipeline:
    def __init__(self, 
                 config, 
                 default_args,
                 vertices: np.array,
                 providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        
        self.PosePipeline = posePipeline(path=config.KPTS_MODEL_PATH, providers=providers)
        self.DetectionTrackingPipeline = detectionTrackingPipeline(default_args, config, providers)
        self.vertices = vertices
    
    def __call__(self, frame: Frame) -> List:
        """
        Main Pipeline
        Args:
            image (Frame): Frame instance

        Returns:
            List: consists of ball_detections, online_players, online_targets, view_transformer
        """
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.PosePipeline, frame, self.vertices)
            future2 = executor.submit(self.DetectionTrackingPipeline, frame)
            view_transformer = future1.result()
            ball_detections, online_players, online_targets = future2.result()
        return ball_detections, online_players, online_targets, view_transformer

    @property
    def BALL_PATHS(self):
        return self.DetectionTrackingPipeline.BALL_PATHS