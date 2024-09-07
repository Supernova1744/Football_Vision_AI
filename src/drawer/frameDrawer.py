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

    def draw_rectangle(self, image, points, player_with_ball, center_point = False):
        # Define the coordinates of the rectangle corners
        top_left = points[0]
        top_right = (points[1][0], points[0][1])
        bottom_left = (points[0][0], points[1][1])
        bottom_right = points[1]

        # Draw only the corners of the rectangle
        corner_length = 20
        color = (0, 255, 0)  # Green color
        thickness = 3

        # Top-left corner
        cv2.line(image, top_left, (top_left[0] + corner_length, top_left[1]), color, thickness)
        cv2.line(image, top_left, (top_left[0], top_left[1] + corner_length), color, thickness)

        # Top-right corner
        cv2.line(image, top_right, (top_right[0] - corner_length, top_right[1]), color, thickness)
        cv2.line(image, top_right, (top_right[0], top_right[1] + corner_length), color, thickness)

        # Bottom-left corner
        cv2.line(image, bottom_left, (bottom_left[0] + corner_length, bottom_left[1]), color, thickness)
        cv2.line(image, bottom_left, (bottom_left[0], bottom_left[1] - corner_length), color, thickness)

        # Bottom-right corner
        cv2.line(image, bottom_right, (bottom_right[0] - corner_length, bottom_right[1]), color, thickness)
        cv2.line(image, bottom_right, (bottom_right[0], bottom_right[1] - corner_length), color, thickness)

        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        if center_point:
            cv2.circle(image, center, radius=5, color=color, thickness=-1)

        text = f'Player with ball: {int(player_with_ball)}'
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0)  # White text
        bg_color = (0, 228, 249)  # Red background

        # Get the text size
        (text_width , text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate the position of the text
        x, y = int(top_left[0]), int(top_left[1]) - 20

        # Draw the background rectangle
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)

        # Put the text on the image
        cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return image

    def draw_on_frame(self, frame: Frame, ball_detections, online_players, ball_points, player_with_ball):
        frame_image = self.draw_triangle(frame.frame_image, ball_detections[:, :4], ball_detections[:, 5], ball_detections[:, 5])
        frame_image = self.draw_ellipse(frame_image, online_players[:, :4], online_players[:, 4], online_players[:, 5])
        frame_image = self.draw_rectangle(frame_image, ball_points, player_with_ball)
        return frame_image