import cv2
import numpy as np

def get_ball_part(frame, ball_point, width, height):
    image_height, image_width, _ = frame.shape
    x, y = ball_point
    
    xmin = np.clip(x - width, 0, image_width).astype(int)
    xmax = np.clip(x + width, 0, image_width).astype(int)
    
    ymin = np.clip(y - height, 0, image_height).astype(int)
    ymax = np.clip(y + height, 0, image_height).astype(int)

    return frame[ymin:ymax, xmin:xmax, :], ((xmin, ymin), (xmax, ymax))

def calc_direction(paths, change_rate=0.1):
    if len(paths) == 1:
        return paths[0]
    elif len(paths) >= 2:
        (x1, y1), (x2, y2) = paths[-2], paths[-1]

        dx, dy = x2 - x1, y2 - y1

        dx *= change_rate
        dy *= change_rate

        return (x2 + dx, y2 + dy)
    else:
        return 640, 640
    
def extract_objects(tlwh, image, resize = None):
    tlwh = np.clip(tlwh, 0, max(image.shape))
    xtl, ytl, width, height = tlwh
    cropped_object = image[int(ytl):int(ytl+height), int(xtl): int(xtl + width), :]
    if not isinstance(resize, type(None)):
        cropped_object = cv2.resize(cropped_object, resize)
    return cropped_object
