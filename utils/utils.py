import cv2
import numpy as np

colors = {
    0: (254, 0, 124), # ball
    1: (76, 76, 255), # goalkeeper
    2: (184, 254, 243), # player
    3: (102, 124, 13), # referee
    4: (0, 175, 255), # player
    5: (0, 0, 255) # player
}

def draw_boxes(image, labels, classes, ids):
    for class_id, label, idx in zip(classes, labels, ids):
        x_center, y_center, width, height = label
        if class_id != 0:
            y_max = y_center + (height / 2)
            cv2.ellipse(
                image,
                center=(int(x_center), int(y_max)),
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=-35,
                endAngle=225,
                color=colors.get(int(class_id), (0,255,0)),
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
                color=colors.get(int(class_id), (0,255,0)),
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
            bg_color = colors.get(int(class_id), (0, 228, 249))   # Red background

            # Get the text size
            (text_width , text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate the position of the text
            x, y = int(x_center) - 20, int(y_max) + 20

            # Draw the background rectangle
            cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)

            # Put the text on the image
            cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        else:
            y_min = y_center - (height / 2)
            x_min = x_center - (width / 2)

            x, y = x_min + 5, y_min - 15  # Change these coordinates to your object's location
            triangle_points = np.array([[x, y + 20], [x - 10, y], [x + 10, y]], np.int32)
            triangle_points2 = np.array([[x, y + 15], [x - 5, y + 3], [x + 5, y + 3]], np.int32)

            triangle_points = triangle_points.reshape((-1, 1, 2))
            triangle_points2 = triangle_points2.reshape((-1, 1, 2))

            # Draw the triangle
            cv2.polylines(image, [triangle_points], isClosed=True, color=colors.get(0, (0,255,0)), thickness=1)
            cv2.polylines(image, [triangle_points2], isClosed=True, color=colors.get(0, (0,255,0)), thickness=1)
            cv2.fillPoly(image, [triangle_points2], color=colors.get(0, (0,255,0)))
    return image

def draw_points(image, kpts, closed=False):
    for x, y in kpts:
        cv2.circle(
            image,
            (int(x), int(y)),
            radius=5, color=(0, 255, 0) if not closed else (255,0,0), thickness=-1
        )
    if closed:
        image = cv2.polylines(image, [kpts.astype(int)], True, (0,0,0), 2)
    return image