import os
import cv2
import numpy as np

class videoWriter:
    def __init__(self, config, stream_hight, stream_width, fps, fourcc, video_name='output.mp4') -> None:
        if not os.path.exists(r".\output"):
            os.mkdir(r".\output")
        self.config = config
        
        self.stream_hight = stream_hight
        self.stream_width = int(stream_width * 1.5)
        
        self.window_height = stream_hight // 2
        self.window_width = stream_width // 2

        self.out = cv2.VideoWriter(r".\output\\" + video_name, fourcc, fps, (self.stream_width, self.stream_hight))
        self.background = np.zeros((self.stream_hight, self.stream_width, 3))
        
    
    def write(self, annotated_frame, pitch, ball_part):
        pitch = cv2.resize(pitch, (self.window_width, self.window_height))
        ball_part = cv2.resize(ball_part, (self.window_width, self.window_height))

        background = self.background.copy().astype(annotated_frame.dtype)
        background += np.array(self.config.BG_COLOR, dtype=annotated_frame.dtype)[np.newaxis, np.newaxis, :]
        background[:, :-self.window_width, :] = annotated_frame
        background[:self.window_height, -self.window_width:, :] = pitch
        background[self.window_height:, -self.window_width:, :] = ball_part
        self.out.write(background)

    def release(self):
        self.out.release()
        