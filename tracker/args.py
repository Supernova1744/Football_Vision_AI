class args:
    # tracking args
    track_high_thresh=0.6
    track_low_thresh=0.1
    new_track_thresh=0.7
    track_buffer=300
    match_thresh=0.8
    aspect_ratio_thresh=1.6
    min_box_area=10
    fuse_score=False

    # CMC
    cmc_method="orb"

    # ReID
    with_reid=False
    proximity_thresh=0.5
    appearance_thresh=0.25
    name=None
    ablation=False
    mot20 = True