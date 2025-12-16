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
    page_title="Helix Detect Pro",
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
        
        if st.button("Analyze Video"):
            st.warning("Video processing can be slow on CPU. Please check console for progress.")
            
            # Simple Frame Processing Loop for Demo
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for performance if needed
                # frame = cv2.resize(frame, (640, 640))
                
                # Inference
                results = model.predict(frame, conf=confidence_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Display
                stframe.image(res_plotted, channels="BGR", use_column_width=True)
            
            cap.release()

with tab3:
    st.markdown("#### 📸 Real-time Webcam Stream")
    
    # Callback for video processing
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        # Note: 'model' is loaded from cache_resource scope. 
        # Inside threads, we might need to be careful, but YOLO is usually valid.
        results = model.predict(img, conf=confidence_threshold, verbose=False)
        res_plotted = results[0].plot()
        
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

    webrtc_streamer(
        key="realtime-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    st.info("Note: Real-time performance depends on your device CPU/GPU capabilities.")
