import streamlit as st
from streamlit_webrtc import webrtc_streamer
from callback import callback
from src.face_detector import FaceDetector
from twilio.rest import Client

account_sid=st.secrets["sid"]
auth_token=st.secrets["token"]
client=Client(account_sid,auth_token)

@st.cache_resource
def get_detector():
    return FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP32/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=0.8,
                            overlap_thr=0.3)


token=client.tokens.create()
st.title("Person Counter App")

show_metrics = st.checkbox("Show FPS and Latency", value=False)

detector=get_detector()


webrtc_streamer(key="sample", video_frame_callback=lambda frame:callback(frame,detector,show_metrics),media_stream_constraints={"video":True,"audio":False},
                rtc_configuration={"iceServers":token.ice_servers})