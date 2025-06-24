import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import cv2
import pandas as pd
from datetime import datetime
from av import VideoFrame

st.title("📦 Real-Time Object Detector (YOLOv5)")
st.markdown("Detecting objects in your webcam feed.")

# UI controls
logging_enabled = st.sidebar.checkbox("✅ Enable Logging", value=True)
st.sidebar.write("Only new, unique detections are logged.")

# Initialize session state to store detections
if "logged_objects" not in st.session_state:
    st.session_state.logged_objects = set()

if "logged_rows" not in st.session_state:
    st.session_state.logged_rows = []

@st.cache_resource
def load_model():
    return torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
model = load_model()

# Load model using local weights
# model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
TARGET_CLASSES = ['book', 'bottle', 'cup', 'person']

# Core detection logic (inlined here)
def detect_objects(frame, logged_objects=set(), logged_rows=[]):
    '''
    - frame: one image/frame from webcam
    - logged_objects: set of already detected object types (to avoid duplicate logs)
    - logged_rows: list of row dicts to log into CSV

    returns:
        - frame (with boxes drawn),
        - updated logged_objects
        - updated logged_rows
    '''

    results = model(frame) # running YOLO on image
    detections = results.pandas().xyxy[0] # converts predictions to pandas df (xyxy gives boxing coordinates)

    for _, row in detections.iterrows(): # go through each detected object
        label = row['name']
        if label in TARGET_CLASSES and label not in logged_objects: # only act if object is in the target classes and hasn't been logged yet
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']]) # boxing coordinates

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # drawing a box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # writing name above box

            logged_objects.add(label)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logged_rows.append({
                "object": label,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "timestamp": timestamp
            })

    return frame, logged_objects, logged_rows

# Streamlit transformer class
class YOLOTransformer(VideoTransformerBase):
    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if logging_enabled:
            img, st.session_state.logged_objects, st.session_state.logged_rows = detect_objects(
                img,
                st.session_state.logged_objects,
                st.session_state.logged_rows
            )
        else:
            img, _, _ = detect_objects(img, set(), [])

        # Convert back to VideoFrame
        return VideoFrame.from_ndarray(img, format="bgr24")
    

# Start video stream
webrtc_streamer(key="yolo", video_processor_factory=YOLOTransformer)

# Show detected objects
if st.session_state.logged_objects:
    st.sidebar.subheader("🔍 Detected Objects")
    for obj in st.session_state.logged_objects:
        st.sidebar.write(f"• {obj}") # write detected objects in sidebar

# Save CSV
if st.session_state.logged_rows:
    df = pd.DataFrame(st.session_state.logged_rows)

    # Create downloadable CSV in memory
    csv_filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv"
    ) # download CSV
