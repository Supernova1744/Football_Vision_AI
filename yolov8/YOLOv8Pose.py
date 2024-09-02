import numpy as np

from yolov8.utils import singleclass_nms
from .YOLOv8 import YOLOv8


class YOLOv8Pose(YOLOv8):
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        super().__init__(path)
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

    def run(self, image):
        input_tensor = self.preprocess(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.kpts = self.postprocess(outputs)
        self.boxes = self.rescale_boxes(self.boxes)
        self.kpts = self.rescale_kpts(self.kpts)

        return self.boxes, self.scores, self.kpts


    def postprocess(self, output):
        predictions = np.squeeze(output[0]).T

        scores = predictions[:, 4]
        # Filter out object confidence scores below threshold
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        boxes = predictions[:, :4]
        kpts = predictions[:, 5:].reshape(-1, 32, 3)

        indecis = singleclass_nms(boxes, scores, self.iou_threshold)
        boxes = boxes[indecis]
        scores = scores[indecis]
        kpts = kpts[indecis]

        return np.squeeze(boxes), np.squeeze(scores), np.squeeze(kpts)

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def rescale_kpts(self, kpts):
        # Rescale kpts to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, 1])
        kpts = np.divide(kpts, input_shape, dtype=np.float32)
        kpts *= np.array([self.img_width, self.img_height, 1])
        return kpts
