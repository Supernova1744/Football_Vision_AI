class config:
    # class ids
    BALL_CLS_ID = 0
    GOALKEEPER_CLS_ID = 1
    PLAYER_CLS_ID = 2
    REFERRE_CLS_ID = 3
    TEAM_ONE_ID = 4
    TEAM_TWO_ID = 5
    COLORS = {
        0: (254, 0, 124), # ball
        1: (76, 76, 255), # goalkeeper
        2: (184, 254, 243), # player
        3: (102, 124, 13), # referee
        4: (0, 175, 255), # player 1
        5: (99, 3, 209) # player 2
    }
    BALL_ICON = r'assets\ball.png'
    TEAM1_ICON = r'assets\player1.png'
    TEAM2_ICON = r'assets\player2.png'
    REFEREE_ICON = r'assets\referee.png'
    KEEPER_ICON = r'assets\keeper.png'

    # detection
    DET_MODEL_PATH = r"resources\weights\yolov8_exp7.onnx"
    DET_CONF_THRESHOLD = 0.25
    DET_IOU_THRESHOLD = 0.6

    # clustering
    TRAIN_THRESHOLD = 10
    REDUCER_NAME = 'pca'
    CLUSTERING_NAME = 'kmeans'
    N_FEATURES = 13
    PLAYER_SIZE = (64, 128)
    CLUSTERING_PIPELINE_PATH = r'resources\sklearn_pipelines\pca_kmeans_pipeline.pkl'
    
    # key points
    KPTS_THRESHOLD = 0.5
    TRAIN_THRESHOLD = 10
    KPTS_MODEL_PATH = r"resources\weights\yolov8_pose.onnx"
    
    # frame
    PADDING = 20
    BG_COLOR = (156, 103, 85)