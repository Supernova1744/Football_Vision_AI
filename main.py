import cv2
import argparse
import numpy as np
from src.utils import get_ball_part
from src.modules.tracker.args import default_args
from src.external import *
from src.modules.videoCapture import videoStreamer, videoWriter
from src.pipelines import mainPipeline
from config.default_config import config
from src.drawer import pitchDrawer, frameDrawer
from src.models.base.serviceProvider import serviceProvider


def main():
    parser = argparse.ArgumentParser(description="Analyze a football match video.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--gpu', action='store_true', required=False, default=False, help='Path to the video file')
    args = parser.parse_args()
    
    VIDEO_PATH = args.video
    GPU = args.gpu

    providers = serviceProvider.CPUExecutionProvider
    if GPU:
        providers = serviceProvider.CUDAExecutionProvider

    VIDEO_NAME = VIDEO_PATH.split("\\")[-1]
    
    video_streamer = videoStreamer(VIDEO_PATH)
    stream_hight, stream_width = video_streamer.video_size

    fps = video_streamer.video_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = videoWriter(config, stream_hight, stream_width, fps, fourcc, VIDEO_NAME)


    PitchConfig = SoccerPitchConfiguration()
    vertices = np.array(PitchConfig.vertices)
    PitchDrawer = pitchDrawer(PitchConfig, config)
    FrameDrawer = frameDrawer(config)
    MainPipeline = mainPipeline(config, default_args, vertices, providers)

    for frame in video_streamer():
        ball_detections, online_players, online_targets, player_with_ball, view_transformer = MainPipeline(frame)
        ball_part, ball_points = get_ball_part(frame.frame_image.copy(), MainPipeline.BALL_PATHS[-1], 70, 70)
        pitch = PitchDrawer.pitch_processing(ball_detections, online_players, online_targets, view_transformer)
        frame_image = FrameDrawer.draw_on_frame(frame, ball_detections, online_players, online_targets, ball_points, player_with_ball)
        
        video_writer.write(frame_image, pitch, ball_part)

    video_writer.out.release()

if __name__ == "__main__":
    main()

    