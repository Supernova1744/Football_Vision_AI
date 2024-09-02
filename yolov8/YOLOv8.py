import cv2
import numpy as np

from yolov8.utils import singleclass_nms
from basemodel import BaseONNXModel


class YOLOv8(BaseONNXModel):
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        super().__init__(path)
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

    def run(self, image):
        input_tensor = self.preprocess(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.postprocess(outputs)

        return self.boxes, self.scores, self.class_ids

    def preprocess(self, image):
        """
            Takes BGR Image
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_tensor = np.ascontiguousarray(input_tensor)

        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def postprocess(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        ball_ids = np.where(class_ids == 0)[0]
        human_ids = np.where(class_ids != 0)[0]

        ball_boxes = boxes[ball_ids]
        ball_cls_ids = class_ids[ball_ids]
        ball_scores = scores[ball_ids]

        human_boxes = boxes[human_ids]
        human_cls_ids = class_ids[human_ids]
        human_scores = scores[human_ids]

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        human_indices = singleclass_nms(human_boxes, human_scores, self.iou_threshold)

        if len(ball_scores):
            ball_indices = [np.argmax(ball_scores)] # Only One ball in the field
        else:
            ball_indices = []
        boxes = np.concatenate([ball_boxes[ball_indices], human_boxes[human_indices]], axis=0)
        scores = np.concatenate([ball_scores[ball_indices], human_scores[human_indices]], axis=0)
        class_ids = np.concatenate([ball_cls_ids[ball_indices], human_cls_ids[human_indices]], axis=0)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
