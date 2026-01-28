import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["absl.logging.min_log_level"] = "2"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


st.set_page_config(
    page_title="AI Speaking Coach",
    layout="wide",
)

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #020024, #090979, #00d4ff);
        color: white !important;
    }

    h1 {
        text-align:center;
        font-size: 40px !important;
        color: white !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.12);
        padding: 18px 20px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        color: white;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 900;
    }

    .section-header {
        font-size: 22px;
        font-weight: 800;
        color: white;
    }

    .feedback-box {
        padding: 16px;
        border-radius: 16px;
        background: rgba(0, 0, 0, 0.25);
        color: white;
        font-size: 18px;
        margin-top: 8px;
    }

    .stImage {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 4px 25px rgba(0,0,0,0.4);
    }

</style>
""", unsafe_allow_html=True)



mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

BLINK_THRESHOLD = 0.21


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)



def get_eye_contact(lm):
    nose = np.array([lm[1].x, lm[1].y])

    left_eye = np.array([lm[33].x, lm[33].y])
    right_eye = np.array([lm[263].x, lm[263].y])

    d_left = np.linalg.norm(nose - left_eye)
    d_right = np.linalg.norm(nose - right_eye)

    if d_right == 0:
        return 0

    ratio = d_left / d_right

    return 1 if 0.85 < ratio < 1.15 else 0



def get_posture_score(pose):
    left_shoulder = pose[11]
    right_shoulder = pose[12]
    nose = pose[0]
    left_hip = pose[23]
    right_hip = pose[24]

    shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
    shoulder_score = 100 if shoulder_slope < 0.01 else 80 if shoulder_slope < 0.03 else 60

    head_tilt = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)
    head_score = 100 if head_tilt < 0.02 else 80 if head_tilt < 0.05 else 60

    spine_slope = abs(left_shoulder.x - left_hip.x) + abs(right_shoulder.x - right_hip.x)
    spine_score = 100 if spine_slope < 0.05 else 80 if spine_slope < 0.1 else 60

    return int((shoulder_score + head_score + spine_score) / 3)


class SpeakingCoach:
    def __init__(self):
        self.eye_frames = 0 #number of frames user was making eye contact
        self.total = 0      #total frames processed
        self.blinks = 0     #total blink count
        self.blink_state = False   # eyes are closed/open true/false
        self.start = time.time()   #timestamp when session started
        self.head = deque(maxlen=50)  #rolling buffer storing the last 50 head positions

        self.face = mp_face_mesh.FaceMesh(  #detects 1 face returns 468 facial landmarks iris/eye area for accuracy
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = mp_pose.Pose(  #This detects the full body skeleton.
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def reset(self):
        self.eye_frames = 0
        self.total = 0
        self.blinks = 0
        self.blink_state = False
        self.start = time.time()
        self.head.clear()

    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = self.face.process(rgb)  #facial landmarks
        pose = self.pose.process(rgb)  #body landmarks

        eye_contact = 0  #initially model dosent detect so eyecontact & posture-->0
        posture = 0

        if face.multi_face_landmarks:
            lm = face.multi_face_landmarks[0].landmark

            left = np.array([[lm[i].x, lm[i].y] for i in LEFT_EYE_IDX])
            right = np.array([[lm[i].x, lm[i].y] for i in RIGHT_EYE_IDX])

            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2

            if ear < BLINK_THRESHOLD and not self.blink_state:
                self.blinks += 1
                self.blink_state = True
            elif ear >= BLINK_THRESHOLD:
                self.blink_state = False

            eye_contact = get_eye_contact(lm)
            self.head.append((lm[1].x, lm[1].y))

        if pose.pose_landmarks:
            posture = get_posture_score(pose.pose_landmarks.landmark)

        self.total += 1
        if eye_contact:
            self.eye_frames += 1

        eye_pct = self.eye_frames / self.total * 100

        if len(self.head) > 10:
            head_score = 100 - np.std(np.array(self.head)) * 700
        else:
            head_score = 100

        head_score = float(np.clip(head_score, 0, 100))

        bpm = self.blinks / max((time.time() - self.start) / 60, 0.01) #Divides blink count by minutes elapsed.
        blink_penalty = max(0, 100 - abs(bpm - 15) * 5) #Ideal blink rate ≈ 15/min far from 15 ->lower score

        conf = (
            0.35 * eye_pct + #eye contact
            0.25 * head_score +
            0.25 * posture +
            0.15 * blink_penalty #Blink Naturalness
        )

        return {
            "eye": eye_pct,  
            "bpm": bpm,  #b;ink per minute
            "posture": posture,
            "head": head_score,
            "conf": float(np.clip(conf, 0, 100))
        }


st.title("🎤 AI Public Speaking & Interview Coach")

with st.sidebar:
    st.header("⚙️ Controls")
    run = st.checkbox("Start Camera")
    reset = st.button("🔄 Reset Session")

if "coach" not in st.session_state:
    st.session_state.coach = SpeakingCoach()

if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

if reset:
    st.session_state.coach.reset()

col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()

with col2:
    st.markdown("<p class='section-header'>📊 Live Metrics</p>", unsafe_allow_html=True)
    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()
    m4 = st.empty()
    conf_box = st.empty()
    fb_box = st.empty()

if run:
    while run:
        ok, frame = st.session_state.cap.read()
        if not ok:
            st.error("Camera not accessible.")
            break

        frame = cv2.flip(frame, 1)
        metrics = st.session_state.coach.run(frame)

        frame_placeholder.image(frame, channels="BGR")

        m1.markdown(f"<div class='metric-card'>👀 Eye Contact<div class='metric-value'>{metrics['eye']:.1f}%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'>✨ Blinks / min<div class='metric-value'>{metrics['bpm']:.1f}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'>🧍 Posture<div class='metric-value'>{int(metrics['posture'])}/100</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'>🧠 Head Stability<div class='metric-value'>{int(metrics['head'])}/100</div></div>", unsafe_allow_html=True)

        conf_box.markdown(
            f"<div class='metric-card'><b>💯 Confidence Score:</b><br><span class='metric-value'>{int(metrics['conf'])}/100</span></div>",
            unsafe_allow_html=True
        )

        fb = []
        if metrics['eye'] < 50:
            fb.append("👀 Increase eye contact with your audience.")
        if metrics['posture'] < 70:
            fb.append("🧍 Sit upright and align your shoulders.")
        if metrics['head'] < 70:
            fb.append("🧠 Try keeping your head steady.")
        if metrics['bpm'] < 10 or metrics['bpm'] > 20:
            fb.append("🙂 Blink naturally — avoid staring.")

        if fb:
            fb_box.markdown("<div class='feedback-box'>⚡ " + " ".join(fb) + "</div>", unsafe_allow_html=True)
        else:
            fb_box.markdown("<div class='feedback-box'>🌟 Excellent delivery! Keep going!</div>", unsafe_allow_html=True)

        time.sleep(0.03)
