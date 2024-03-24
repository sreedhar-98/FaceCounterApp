import time
from src import utils
import av
import math


n_frames = 0
fps_cum = 0.0
fps_avg = 0.0

def callback(frame,detector,show_metrics):
    global n_frames,fps_cum,fps_avg
    start_time = time.perf_counter()
    n_frames+=1
    frame=frame.to_ndarray(format="bgr24")
    bboxes, scores = detector.inference(frame)
    frame = utils.draw_boxes(frame, bboxes)
    end_time = time.perf_counter()
    fps = 1.0 / (end_time - start_time)
    fps_cum += fps
    fps_avg = fps_cum / n_frames
    latency=end_time-start_time
    if show_metrics:
        frame = utils.put_text_on_image(frame, text='FPS: {}'.format(math.ceil(fps_avg)),position=(10,90))
        frame=utils.put_text_on_image(frame,text='Latency : {} ms'.format(math.ceil(latency*1000)),position=(10,130))
    return av.VideoFrame.from_ndarray(frame,format="bgr24")