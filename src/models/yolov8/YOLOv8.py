import cv2
import numpy as np
from typing import List
from .utils import multiclass_nms
from ..base import BaseONNXModel, serviceProvider

class YOLOv8(BaseONNXModel):
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, providers: serviceProvider = serviceProvider.CPUExecutionProvider):
        super().__init__(path, providers)
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

    def run(self, image: np.array):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.postprocess(outputs)
        return self.boxes, self.scores, self.class_ids

    def preprocess(self, image: np.array) -> np.array:
        """
            Takes BGR Image
        """
        self.img_height, self.img_width = image.shape[:2]
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        input_tensor = image[np.newaxis, :, :, :]
        input_tensor = np.ascontiguousarray(input_tensor)
        return input_tensor

    def postprocess(self, output: np.array) -> List:
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        selected_indecis = scores > self.conf_threshold

        predictions = predictions[selected_indecis, :]
        scores = scores[selected_indecis]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)

        ball_ids = np.where(class_ids == 0)[0]
        human_ids = np.where(class_ids != 0)[0]

        ball_boxes = boxes[ball_ids]
        ball_cls_ids = class_ids[ball_ids]
        ball_scores = scores[ball_ids]

        human_boxes = boxes[human_ids]
        human_cls_ids = class_ids[human_ids]
        human_scores = scores[human_ids]

        human_indices = multiclass_nms(human_boxes, human_scores, human_cls_ids, self.iou_threshold)

        if len(ball_scores):
            ball_indices = [np.argmax(ball_scores)] # Only One ball in the field
        else:
            ball_indices = []
        boxes = np.concatenate([ball_boxes[ball_indices], human_boxes[human_indices]], axis=0)
        scores = np.concatenate([ball_scores[ball_indices], human_scores[human_indices]], axis=0)
        class_ids = np.concatenate([ball_cls_ids[ball_indices], human_cls_ids[human_indices]], axis=0)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
