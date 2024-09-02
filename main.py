import os
import cv2
import joblib
import argparse
import numpy as np

from collections import Counter
from utils import draw_boxes, colors
from yolov8.utils import xywh2xyxy
from yolov8 import YOLOv8, YOLOv8Pose

from tracker.args import args
from tracker.mc_bot_sort import BoTSORT

from external import *

CONFIG = SoccerPitchConfiguration()

DETECTOR = YOLOv8(
    path=r"weights\yolov8_exp7.onnx",
    conf_thres=0.25,
    iou_thres=0.6
)

POSE_EST = YOLOv8Pose(
    path=r"weights\yolov8_pose.onnx"
    )

TRACKER = BoTSORT(
args=args, frame_rate=30
)

CLUSTERING = joblib.load(
    filename=r'pipelines\pca_kmeans_pipeline.pkl'
    )

REDUCER_NAME = 'pca'
CLUSTERING_NAME = 'kmeans'

BALL_CLS_ID = 0
GOALKEEPER_CLS_ID = 1
PLAYER_CLS_ID = 2
REFERRE_CLS_ID = 3
TEAM_ONE_ID = 4
TEAM_TWO_ID = 5

FLAG = 0
KPTS_THRESHOLD = 0.5
TRAIN_THRESHOLD = 10
PLAYER_SIZE = (64, 128) # [W, H]
MATCH_MAP = {}

vertices = np.array(CONFIG.vertices)

def extract_objects(tlwh, image):
    tlwh = np.clip(tlwh, 0, max(image.shape))
    xtl, ytl, width, height = tlwh
    cropped_object = image[int(ytl):int(ytl+height), int(xtl): int(xtl + width), :]
    return cropped_object

def class_update(online_targets):
    for target in online_targets:
        cls = target[4]
        track_id = target[5]
        class_history = MATCH_MAP.get(track_id, [])
        if len(class_history):
            counter = Counter(class_history)
            most_common_element, _ = counter.most_common(1)[0]
            target[4] = most_common_element
        class_history.append(cls)
        MATCH_MAP[track_id] = class_history
    return online_targets

def process_target(target):
    return [*target.tlwh, target.cls, target.track_id, target.score, target.tlwh[2] * target.tlwh[3]]

def Pipeline(image):
    global FLAG
    global KPTS_THRESHOLD

    # KeyPoints Detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, _, kpts = POSE_EST(image=image.copy())
    kpts_indecis = kpts[:, -1] >= KPTS_THRESHOLD
    kpts = kpts[kpts_indecis, :]
    pitch_points = vertices[kpts_indecis]

    # Object Detection
    boxes, scores, class_ids = DETECTOR(image.copy())
    xyxy = xywh2xyxy(boxes)

    # Prepare For Tracking
    human_ids = np.where(class_ids != BALL_CLS_ID)
    ball_ids  = np.where(class_ids == BALL_CLS_ID)

    # Handle Ball Detection
    ball_detections = np.concatenate([boxes, scores[:, np.newaxis], class_ids[:, np.newaxis]], axis=-1)[ball_ids]
    if ball_detections.shape[0]:
        ball_detections = ball_detections[np.argmax(ball_detections[:, 4])][np.newaxis, :]
    
    # Concatenate data for Tracker & Run Tracker on (HUMAN ONLY) data
    human_detections = np.concatenate([xyxy, scores[:, np.newaxis], class_ids[:, np.newaxis]], axis=-1)[human_ids]
    online_targets = TRACKER.update(human_detections.astype(np.float64), image.copy())
    del human_detections

    # Handle Tracker output
    online_targets = [[*target.tlwh, target.cls, target.track_id, target.score, target.tlwh[2] * target.tlwh[3]] for target in online_targets]
    online_targets = np.array(online_targets)
    online_targets = class_update(online_targets)

    # Divide data into players and non-players and filter small detections
    online_targets = online_targets[np.where(online_targets[:, -1] > args.min_box_area)]
    online_players = online_targets[np.where(online_targets[:, 4] == PLAYER_CLS_ID)]
    online_targets = online_targets[np.where(online_targets[:, 4] != PLAYER_CLS_ID)]

    # Crop players images for ReID and Clustering
    cropped_players = [extract_objects(player[:4], image.copy()) for player in online_players]

    # Handle Player Clustering
    resized_playes = [cv2.resize(pimg, PLAYER_SIZE).reshape(-1, 1).squeeze() for pimg in cropped_players]

    # Run Dim-Reduction and Clustering [Fit for first 10 times]
    if FLAG < TRAIN_THRESHOLD and len(resized_playes) >= 13:
        transformed_data = CLUSTERING.named_steps[REDUCER_NAME].fit_transform(resized_playes)
        cluster_label = CLUSTERING.named_steps[CLUSTERING_NAME].fit_predict(transformed_data)
        FLAG += 1
    else:
        cluster_label = CLUSTERING.predict(resized_playes)
    
    # Overwrite player ids
    online_players[:, 4] = np.where(cluster_label == 0, TEAM_ONE_ID, TEAM_TWO_ID)
    
    del resized_playes
    del cluster_label

    # TLWH 2 XYWH
    online_players[:, 0] = online_players[:, 0] + (online_players[:, 2] / 2)
    online_players[:, 1] = online_players[:, 1] + (online_players[:, 3] / 2)

    online_targets[:, 0] = online_targets[:, 0] + (online_targets[:, 2] / 2)
    online_targets[:, 1] = online_targets[:, 1] + (online_targets[:, 3] / 2)

    return ball_detections, online_players, online_targets, kpts, pitch_points
    
def get_frame(cap, stop_frame_id=-1):
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if frame_id == stop_frame_id:
            return
        if ret:
            frame_id += 1
            yield frame, frame_id
        else:
            return

def draw_on_frame(frame, ball_detections, online_players, online_targets, kpts, pitch_points):
    pitch = draw_pitch(CONFIG)
    view_transformer = ViewTransformer(kpts[:, :2], pitch_points)

    pitch_team_1 = view_transformer.transform_points(online_players[online_players[:, 4] == TEAM_ONE_ID, :2])
    pitch_team_2 = view_transformer.transform_points(online_players[online_players[:, 4] == TEAM_TWO_ID, :2])
    pitch_goalkeeper = view_transformer.transform_points(online_targets[online_targets[:, 4] == GOALKEEPER_CLS_ID, :2])
    pitch_referres = view_transformer.transform_points(online_targets[online_targets[:, 4] == REFERRE_CLS_ID, :2])
    
    pitch = draw_pitch_voronoi_diagram(CONFIG, pitch_team_1, pitch_team_2, colors[TEAM_ONE_ID][::-1], colors[TEAM_TWO_ID][::-1], pitch=pitch, opacity=0.4)

    
    pitch = draw_points_on_pitch(CONFIG,
                            pitch_referres,
                            pitch=pitch,
                            face_color=colors[REFERRE_CLS_ID])
    
    pitch = draw_points_on_pitch(CONFIG,
                            pitch_goalkeeper,
                            pitch=pitch,
                            face_color=colors[GOALKEEPER_CLS_ID])
    
    pitch = draw_points_on_pitch(CONFIG,
                            pitch_team_1,
                            pitch=pitch,
                            face_color=colors[TEAM_ONE_ID][::-1])
    
    pitch = draw_points_on_pitch(CONFIG,
                            pitch_team_2,
                            pitch=pitch,
                            face_color=colors[TEAM_TWO_ID][::-1])

    if ball_detections.shape[0]:
        pitch_ball = view_transformer.transform_points(ball_detections[:, :2])
        pitch = draw_points_on_pitch(CONFIG,
                            pitch_ball,
                            pitch=pitch,
                            face_color=colors[BALL_CLS_ID])
        
    frame = draw_boxes(frame, ball_detections[:, :4], ball_detections[:, 5], ball_detections[:, 5])
    frame = draw_boxes(frame, online_players[:, :4], online_players[:, 4], online_players[:, 5])
    frame = draw_boxes(frame, online_targets[:, :4], online_targets[:, 4], online_targets[:, 5])

    pitch_width = frame.shape[1] // 3
    pitch_height = (pitch_width * frame.shape[0]) // frame.shape[1]
    pitch = cv2.resize(pitch, (pitch_width, pitch_height))
    alpha = 0.6
    roi = frame[-pitch_height:, pitch_width:2*pitch_width, :]
    # Blend the overlay image with the ROI
    blended = cv2.addWeighted(roi, 1 - alpha, pitch, alpha, 0)

    # Place the blended image back into the background image
    frame[-pitch_height:, pitch_width:2*pitch_width, :] = blended
    return frame


def main():
    parser = argparse.ArgumentParser(description="Analyze a football match video.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    
    args = parser.parse_args()
    
    VIDEO_PATH = args.video
    VIDEO_NAME = VIDEO_PATH.split("\\")[-1]
    if not os.path.exists(r".\output"):
        os.mkdir(r".\output")

    cap = cv2.VideoCapture(VIDEO_PATH)
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(r".\output\\" + VIDEO_NAME, fourcc, fps, (iw, ih))
    for frame, frame_id in get_frame(cap):
        print(f"Processing Frame: #{frame_id}")
        ball_detections, online_players, online_targets, kpts, pitch_points = Pipeline(frame)
        frame = draw_on_frame(frame, ball_detections, online_players, online_targets, kpts, pitch_points)
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()

    