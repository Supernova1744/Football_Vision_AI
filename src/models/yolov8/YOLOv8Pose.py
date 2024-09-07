import numpy as np
from typing import List, Union
from .utils import singleclass_nms
from .YOLOv8 import YOLOv8
from ..base import serviceProvider


class YOLOv8Pose(YOLOv8):
    def __init__(self, path: str, 
                 conf_thres: float = 0.7, 
                 iou_thres: float = 0.5, 
                 n_kpts: int = 32, 
                 providers: serviceProvider = serviceProvider.CPUExecutionProvider) -> None:
        super().__init__(path, providers)
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.n_kpts = n_kpts


    def run(self, image: np.array) -> List:
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        boxes, scores, kpts = self.postprocess(outputs)
        return boxes, scores, kpts


    def postprocess(self, output: Union[List, np.array]) -> List:
        predictions = np.squeeze(output[0]).T

        scores = predictions[:, 4]
        selected_indecis = scores > self.conf_threshold
        predictions = predictions[selected_indecis, :]
        scores = scores[selected_indecis]
        boxes = predictions[:, :4]
        kpts = predictions[:, 5:].reshape(-1, self.n_kpts, 3)

        indecis = singleclass_nms(boxes, scores, self.iou_threshold)

        boxes = np.squeeze(boxes[indecis])
        scores = np.squeeze(scores[indecis])
        kpts = np.squeeze(kpts[indecis])

        boxes = self.rescale_boxes(boxes)
        kpts = self.rescale_kpts(kpts)

        return boxes, scores, kpts
    
    def rescale_kpts(self, kpts: np.array) -> np.array:
        input_shape = np.array([self.input_width, self.input_height, 1])
        kpts = np.divide(kpts, input_shape, dtype=np.float32)
        kpts *= np.array([self.img_width, self.img_height, 1])
        return kpts
