import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
import PIL.Image
from streamlit_webrtc import webrtc_streamer
import av

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="HelmGuard Vision",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS for Premium Look
# -------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        h1 {
            color: #ffffff;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        .stButton>button {
            background-color: #00ADB5;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #00FFF5;
            color: #222831;
        }
        .metric-card {
            background-color: #222831;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
        }
        .highlight-text {
            color: #00ADB5;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
model_type = st.sidebar.radio("Model Version", ["Nano (v8n)", "Small (v8s)"], index=0)
st.sidebar.info("Using 'Nano' for maximum speed.")

# -------------------------
# Main Content
# -------------------------
st.title("🛡️ Helix Detect Pro")
st.markdown("### Advanced Real-time Helmet Detection System")
st.markdown("Upload an image or video to detect safety helmet compliance with high precision.")

# Load Model (Cached)
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    # TRY loading the best weights from training
    model_path = 'weights/best.pt'
    model = load_model(model_path)
    st.sidebar.success(f"Model Loaded: {model_path}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Falling back to standard yolov8n.pt for demonstration")
    model = load_model("yolov8n.pt")

# Tabs for different modes
tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "📹 Video Analysis", "📸 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Input")
            image = PIL.Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
        with col2:
            st.markdown("#### Detection Result")
            if st.button("Analyze Image"):
                with st.spinner('Running AI Inference...'):
                    # Perform inference
                    start_time = time.time()
                    results = model.predict(image, conf=confidence_threshold, iou=iou_threshold)
                    end_time = time.time()
                    
                    # Plot results
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption=f"Inference Time: {(end_time-start_time)*1000:.2f}ms", use_column_width=True)
                    
                    # Statistics
                    boxes = results[0].boxes
                    num_helmets = 0
                    num_no_helmets = 0
                    
                    # Assuming class 0 is Helmet and 1 is No-Helmet (Check data.yaml for mapping if unsure)
                    # Based on your training scan: ['With Helmet', 'Without Helmet'] -> 0: With, 1: Without
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:
                            num_helmets += 1
                        else:
                            num_no_helmets += 1
                            
                    st.success(f"✅ With Helmet: {num_helmets}")
                    if num_no_helmets > 0:
                        st.error(f"⚠️ Without Helmet: {num_no_helmets}")
                    else:
                        st.info("No violations detected.")

with tab2:
    uploaded_video = st.file_uploader("Upload a video...", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        if st.button("Generate Analyzed Video"):
            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("Initializing video processing...")
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Codecs to try in order of preference for browser compatibility
            # 1. h264 (avc1) -> Standard .mp4
            # 2. VP80 (vp80) -> WebM (Very reliable on Linux/Web)
            # 3. mp4v -> Legacy .mp4
            codecs = [
                ('avc1', '.mp4'),
                ('vp80', '.webm'),
                ('mp4v', '.mp4')
            ]
            
            output_file_name = ""
            out = None
            used_codec = None
            
            for c, ext in codecs:
                try:
                    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    output_file_name = tfile_out.name
                    tfile_out.close()
                    
                    fourcc = cv2.VideoWriter_fourcc(*c)
                    out_temp = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
                    
                    if out_temp.isOpened():
                        out = out_temp
                        used_codec = c
                        break
                except Exception as e:
                    continue
            
            if not out or not out.isOpened():
                st.error("Could not initialize video writer with any supported codec (H.264/VP8).")
                st.stop()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference
                results = model.predict(frame, conf=confidence_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Write frame
                out.write(res_plotted)
                
                # Progress
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            out.release()
            
            status_text.success(f"Processing Complete using codec: {used_codec}! 🎉")
            progress_bar.empty()
            
            # Display Result
            st.markdown(f"### 🎬 Analyzed Video ({used_codec})")
            st.video(output_file_name)

with tab3:
    st.markdown("#### 📸 Real-time Webcam Stream")
    
    # helper class to persist callback identity
    class YOLOVideoProcessor:
        def __init__(self):
            # These will be updated by the main script
            self.model = None
            self.conf = 0.4
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Use the injected model and settings
            if self.model is not None:
                results = self.model.predict(img, conf=self.conf, verbose=False)
                res_plotted = results[0].plot()
                return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")
            
            return frame

    # Initialize processor in session state if not present
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = YOLOVideoProcessor()

    # Update processor with current controls
    st.session_state.video_processor.model = model
    st.session_state.video_processor.conf = confidence_threshold

    webrtc_streamer(
        key="realtime-detection",
        video_frame_callback=st.session_state.video_processor.recv,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    st.info("Note: Real-time performance depends on your device CPU/GPU capabilities.")
