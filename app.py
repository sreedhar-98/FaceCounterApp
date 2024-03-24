import streamlit as st
from streamlit_webrtc import webrtc_streamer
from callback import callback
from src.face_detector import FaceDetector
from twilio.rest import Client
import logging


@st.cache_resource()
def get_detector():
    logging.info("Detector Created")
    return FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP32/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=0.8,
                            overlap_thr=0.3)

@st.cache_data(ttl=86400)
def get_twilio_token():
    logging.info("Twilio Details accesed")
    account_sid=st.secrets["sid"]
    auth_token=st.secrets["token"]
    client=Client(account_sid,auth_token)
    token=client.tokens.create()
    return token


token=get_twilio_token()
logging.info("User called")

st.title("Person Counter App")

show_metrics = st.checkbox("Show FPS and Latency", value=False)

detector=get_detector()


webrtc_streamer(key="sample", video_frame_callback=lambda frame:callback(frame,detector,show_metrics),media_stream_constraints={"video":True,"audio":False},
                 rtc_configuration={"iceServers":token.ice_servers})