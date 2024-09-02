import sys
sys.path.append(".")

import cv2
import tqdm
import joblib
import numpy as np

from yolov8 import YOLOv8
from yolov8.utils import xywh2xyxy

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


DETECTOR = YOLOv8(
    path=r"weights\yolov8_exp7.onnx",
    conf_thres=0.25,
    iou_thres=0.6
)

REDUCER = PCA(n_components=13)
CLUSTERING_MODEL = KMeans(n_clusters=2)
TRAINING_VIDEO_PATH = None #TODO: ADD TRAINING VIDEO PATH
PLAYER_SIZE = (64, 128) # [W, H]


cap = cv2.VideoCapture(TRAINING_VIDEO_PATH)
ret, frame = cap.read()

ALL_IMAGES = []
frame_idx = 0
while ret:
    if frame_idx % 3 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ALL_IMAGES.append(frame)
    ret, frame = cap.read()
    frame_idx += 1
cap.release()

players_features = []
imgs = []


for image in tqdm.tqdm(ALL_IMAGES, desc=f"total number of images is {len(ALL_IMAGES)}"):
    boxes, scores, class_ids = DETECTOR(image)
    xyxy = xywh2xyxy(boxes)

    for (x1,y1,x2,y2) in xyxy[np.where(class_ids == 2)[0]].astype(int):
        player_img = image[y1:y2, x1:x2, :]
        player_img = cv2.resize(player_img, PLAYER_SIZE)
        players_features.append(player_img.reshape(1, -1))
    

players_features = np.array(players_features).squeeze()

predictions = REDUCER.fit_transform(players_features)
clusters = CLUSTERING_MODEL.fit_predict(predictions)


pipeline = Pipeline([
    ('pca', REDUCER),
    ('kmeans', CLUSTERING_MODEL)
])

joblib.dump(pipeline, r'pipelines\pca_kmeans_pipeline_.pkl')
