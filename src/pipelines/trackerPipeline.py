import joblib
import numpy as np
from collections import Counter
from src.utils import extract_objects
from src.modules.tracker.mc_bot_sort import BoTSORT
from src.modules.videoCapture import Frame

class trackerPipeline:
    def __init__(self, args, config, frame_rate=30) -> None:
        self.tracker = BoTSORT(
        args=args, frame_rate=frame_rate
        )
        self.clustering = joblib.load(filename=config.CLUSTERING_PIPELINE_PATH)

        self.MATCH_MAP = dict()
        self.args = args
        self.config = config
        self.FLAG = 0
        

    def class_update(self, online_targets):
        for target in online_targets:
            cls = target[4]
            track_id = target[5]
            class_history = self.MATCH_MAP.get(track_id, [])
            if len(class_history):
                counter = Counter(class_history)
                most_common_element, _ = counter.most_common(1)[0]
                target[4] = most_common_element
            class_history.append(cls)
            self.MATCH_MAP[track_id] = class_history
        return online_targets

    def __call__(self, frame: Frame, human_detections: np.array):
        global FLAG
        online_targets = self.tracker.update(human_detections.astype(np.float64), frame.rgb_image)
        del human_detections

        # Handle Tracker output
        online_targets = [[*target.tlwh, target.cls, target.track_id, target.score, target.tlwh[2] * target.tlwh[3]] for target in online_targets]
        online_targets = np.array(online_targets)
        online_targets = self.class_update(online_targets)

        # Divide data into players and non-players and filter small detections
        online_targets = online_targets[np.where(online_targets[:, -1] > self.args.min_box_area)]
        online_players = online_targets[np.where(online_targets[:, 4] == self.config.PLAYER_CLS_ID)]
        online_targets = online_targets[np.where(online_targets[:, 4] != self.config.PLAYER_CLS_ID)]

        # Crop players images for ReID and Clustering
        resized_playes = [extract_objects(player[:4], frame.rgb_image, self.config.PLAYER_SIZE).reshape(-1, 1).squeeze() for player in online_players]

        # Run Dim-Reduction and Clustering [Fit for first 10 times]
        if self.FLAG < self.config.TRAIN_THRESHOLD and len(resized_playes) >= self.config.N_FEATURES:
            transformed_data = self.clustering.named_steps[self.config.REDUCER_NAME].fit_transform(resized_playes)
            cluster_label = self.clustering.named_steps[self.config.CLUSTERING_NAME].fit_predict(transformed_data)
            self.FLAG += 1
        else:
            cluster_label = self.clustering.predict(resized_playes)
        
        # Overwrite player ids
        online_players[:, 4] = np.where(cluster_label == 0, self.config.TEAM_ONE_ID, self.config.TEAM_TWO_ID)
        
        del resized_playes
        del cluster_label

        # TLWH 2 XYWH
        online_players[:, 0] = online_players[:, 0] + (online_players[:, 2] / 2)
        online_players[:, 1] = online_players[:, 1] + (online_players[:, 3] / 2)

        online_targets[:, 0] = online_targets[:, 0] + (online_targets[:, 2] / 2)
        online_targets[:, 1] = online_targets[:, 1] + (online_targets[:, 3] / 2)

        return online_players, online_targets
