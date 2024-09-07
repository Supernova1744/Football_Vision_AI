import cv2
import numpy as np
from src.modules.videoCapture import Frame

class frameDrawer:
    def __init__(self, config) -> None:
        self.config = config

    def draw_ellipse(self, image, labels, classes, ids):
        for class_id, label, idx in zip(classes, labels, ids):
            x_center, y_center, width, height = label
            y_max = y_center + (height / 2)
            cv2.ellipse(
                image,
                center=(int(x_center), int(y_max)),
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=-35,
                endAngle=225,
                color=self.config.COLORS.get(int(class_id), (0,255,0)),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                image,
                center=(int(x_center), int(y_max)),
                axes=(int(width) + 3, int(0.35 * width) + 3),
                angle=0.0,
                startAngle=-65,
                endAngle=265,
                color=self.config.COLORS.get(int(class_id), (0,255,0)),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # Annotate the object
            # Define the text and its properties
            text = f'#{int(idx)}'
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 0.6
            font_thickness = 1
            text_color = (0, 0, 0)  # White text
            bg_color = self.config.COLORS.get(int(class_id), (0, 228, 249))   # Red background

            # Get the text size
            (text_width , text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate the position of the text
            x, y = int(x_center) - 20, int(y_max) + 20

            # Draw the background rectangle
            cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)

            # Put the text on the image
            cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return image
    
    def draw_triangle(self,  image, labels, classes, ids):
        for class_id, label, idx in zip(classes, labels, ids):
            x_center, y_center, width, height = label
            y_max = y_center + (height / 2)
            y_min = y_center - (height / 2)
            x_min = x_center - (width / 2)

            x, y = x_min + 5, y_min - 15  # Change these coordinates to your object's location
            triangle_points = np.array([[x, y + 20], [x - 10, y], [x + 10, y]], np.int32)
            triangle_points2 = np.array([[x, y + 15], [x - 5, y + 3], [x + 5, y + 3]], np.int32)

            triangle_points = triangle_points.reshape((-1, 1, 2))
            triangle_points2 = triangle_points2.reshape((-1, 1, 2))

            # Draw the triangle
            cv2.polylines(image, [triangle_points], isClosed=True, color=self.config.COLORS.get(0, (0,255,0)), thickness=1)
            cv2.polylines(image, [triangle_points2], isClosed=True, color=self.config.COLORS.get(0, (0,255,0)), thickness=1)
            cv2.fillPoly(image, [triangle_points2], color=self.config.COLORS.get(0, (0,255,0)))
        return image

    def draw_on_frame(self, frame: Frame, ball_detections, online_players, online_targets, view_transformer, stream_size):
        frame_image = self.draw_triangle(frame.frame_image, ball_detections[:, :4], ball_detections[:, 5], ball_detections[:, 5])
        frame_image = self.draw_ellipse(frame_image, online_players[:, :4], online_players[:, 4], online_players[:, 5])
        return frame_image