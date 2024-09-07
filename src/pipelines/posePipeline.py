import numpy as np
from typing import List
from src.models import YOLOv8Pose
from src.external import ViewTransformer
from src.modules.videoCapture import Frame
from src.models.base.serviceProvider import serviceProvider

class posePipeline:
    def __init__(self, path: str, 
                 conf_thres: float = 0.7, 
                 iou_thres: float = 0.5, 
                 n_kpts: int = 32,
                 kpts_threshold: float = 0.5,
                 providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        
        self.model = YOLOv8Pose(path, conf_thres, iou_thres, n_kpts, providers)
        self.kpts_threshold = kpts_threshold
    
    def __call__(self, frame: Frame, vertices: np.array) -> List:
        """
        Pose Detection Pipeline
        Args:
            image (Frame): Frame instance

        Returns:
            List: consists of two elements <kpts: keypoints, pitch_points: pitch points for each kpts>
        """
        _, _, kpts = self.model(image=frame.rgb_image)
        kpts_indecis = kpts[:, -1] >= self.kpts_threshold
        kpts = kpts[kpts_indecis, :]
        pitch_points = vertices[kpts_indecis]
        view_transformer = ViewTransformer(kpts[:, :2], pitch_points)
        return view_transformer