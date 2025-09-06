import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io
import cv2
import time
from datetime import datetime
from pathlib import Path
import threading
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("YOLO not installed. Install with: pip install ultralytics")

# ==========================
# OPTIMIZED CONFIG
# ==========================
st.set_page_config(page_title="Real-Time Car Damage Detection", layout="wide", page_icon="🚗")

# Performance settings - HEAVILY OPTIMIZED
WEIGHTS_FILE = "best.pt"
FIXED_CONF = 0.25      # Increased confidence (less false positives = faster)
FIXED_IOU = 0.7        
FIXED_IMGSZ = 416      # Reduced from 640 to 416 (much faster!)

# Real-time optimization
FPS_TARGET = 5         # Reduced from 10 to 5 FPS
FRAME_SKIP = 5         # Increased skip (process every 5th frame)
MAX_DETECTIONS = 50    # Limit stored detections
RESIZE_FACTOR = 0.5    # Resize input frames

# Severity thresholds
SEVERITY_T1 = 0.25
SEVERITY_T2 = 0.60

# ==========================
# OPTIMIZED SESSION STATE
# ==========================
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = []
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ==========================
# OPTIMIZED MODEL LOADING
# ==========================
@st.cache_resource(show_spinner=True)
def load_model_optimized():
    """Load YOLO model with optimizations."""
    if not YOLO_AVAILABLE:
        return None
        
    try:
        if Path(WEIGHTS_FILE).exists():
            model = YOLO(WEIGHTS_FILE)
            # Optimize model for inference
            model.fuse()  # Fuse conv and bn layers
            return model
        else:
            st.error(f"Model file {WEIGHTS_FILE} not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================
# SUPER FAST INFERENCE
# ==========================
def run_inference_fast(model, frame_bgr):
    """Ultra-optimized inference for real-time."""
    start_time = time.time()
    
    try:
        # Resize frame for faster processing
        h, w = frame_bgr.shape[:2]
        new_h, new_w = int(h * RESIZE_FACTOR), int(w * RESIZE_FACTOR)
        small_frame = cv2.resize(frame_bgr, (new_w, new_h))
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Fast inference with minimal settings
        results = model.predict(
            source=frame_rgb,
            conf=FIXED_CONF,
            iou=FIXED_IOU,
            imgsz=FIXED_IMGSZ,
            verbose=False,
            save=False,
            show=False,
            stream=False,
            half=True  # Use FP16 for speed
        )
        
        r = results[0]
        detections = []
        annotated_frame = frame_bgr.copy()
        
        # Quick detection extraction
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            
            # Scale boxes back to original size
            scale_x = w / new_w
            scale_y = h / new_h
            
            for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
                # Scale coordinates
                x1, y1, x2, y2 = box
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                cls_name = names.get(cls_id, str(cls_id))
                
                # Simple severity calculation
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < 5000:
                    severity = "Light"
                elif bbox_area < 15000:
                    severity = "Medium"
                else:
                    severity = "Heavy"
                
                # Draw simple rectangle (fastest)
                color = (0, 255, 0) if severity == "Light" else (0, 255, 255) if severity == "Medium" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Simple text
                label = f"{cls_name}"
                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                detections.append({
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "severity": severity,
                    "bbox": [x1, y1, x2, y2]
                })
        
        processing_time = time.time() - start_time
        st.session_state.processing_time = processing_time
        
        return annotated_frame, detections
        
    except Exception as e:
        st.session_state.processing_time = time.time() - start_time
        return frame_bgr, []

# ==========================
# LIGHTWEIGHT VIDEO PROCESSOR
# ==========================
class FastVideoProcessor:
    def __init__(self, model):
        self.model = model
        self.frame_count = 0
        self.last_process_time = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Skip frames for performance
        current_time = time.time()
        if (self.frame_count % FRAME_SKIP == 0 and 
            current_time - self.last_process_time > 1.0/FPS_TARGET):
            
            if self.model is not None:
                annotated_img, detections = run_inference_fast(self.model, img)
                
                # Update detections (limit storage)
                st.session_state.last_detections = detections
                
                if detections:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    for det in detections:
                        det['timestamp'] = timestamp
                    
                    # Limit stored results
                    st.session_state.detection_results.extend(detections)
                    if len(st.session_state.detection_results) > MAX_DETECTIONS:
                        st.session_state.detection_results = st.session_state.detection_results[-MAX_DETECTIONS:]
                
                img = annotated_img
                self.last_process_time = current_time
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================
# LOAD MODEL
# ==========================
st.title("🚗 Fast Real-Time Car Damage Detection")

with st.spinner("Loading AI Model..."):
    model = load_model_optimized()

if model is None:
    st.error("❌ Model tidak tersedia!")
    st.info("Pastikan file 'best.pt' ada di folder aplikasi")
    st.stop()

st.success(f"✅ Model loaded: {WEIGHTS_FILE}")
st.session_state.model_loaded = True

# ==========================
# PERFORMANCE SETTINGS DISPLAY
# ==========================
with st.expander("⚡ Performance Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Size", f"{FIXED_IMGSZ}px")
        st.metric("Frame Skip", f"1/{FRAME_SKIP}")
    with col2:
        st.metric("Target FPS", FPS_TARGET)
        st.metric("Resize Factor", f"{RESIZE_FACTOR}x")
    with col3:
        st.metric("Confidence", f"{FIXED_CONF}")
        st.metric("Max Storage", MAX_DETECTIONS)

# ==========================
# MAIN INTERFACE
# ==========================
st.header("📹 Live Detection")

# WebRTC with minimal config
rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="fast-car-damage",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: FastVideoProcessor(model),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.subheader("🔍 Status")
    
    # Performance metrics
    if st.session_state.processing_time > 0:
        fps_estimate = 1.0 / st.session_state.processing_time
        st.metric("Processing FPS", f"{fps_estimate:.1f}")
    
    # Detection status
    status_placeholder = st.empty()
    if st.session_state.last_detections:
        status_placeholder.success(f"🚨 {len(st.session_state.last_detections)} Found!")
        for det in st.session_state.last_detections[:3]:  # Show max 3
            severity_color = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
            st.write(f"{severity_color} **{det['class_name']}**")
    else:
        status_placeholder.info("👀 Monitoring...")
    
    # Quick stats
    if st.session_state.detection_results:
        st.metric("Total Found", len(st.session_state.detection_results))
    
    # Controls
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.detection_results = []
        st.session_state.last_detections = []
        st.rerun()

# ==========================
# LEGEND
# ==========================
st.subheader("📖 Legend")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Severity:**
    - 🟢 Light: Small damage
    - 🟡 Medium: Moderate damage  
    - 🔴 Heavy: Severe damage
    """)

with col2:
    st.markdown("""
    **Optimizations:**
    - Reduced image size (416px)
    - Process every 5th frame
    - FP16 inference for speed
    """)

# ==========================
# RESULTS SECTION
# ==========================
if st.session_state.detection_results:
    with st.expander("📊 Results", expanded=False):
        df = pd.DataFrame(st.session_state.detection_results)
        st.dataframe(df[['timestamp', 'class_name', 'severity', 'confidence']], use_container_width=True)
        
        # Quick export
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            f"detections_{datetime.now().strftime('%H%M%S')}.csv",
            "text/csv"
        )

# ==========================
# FOOTER
# ==========================
st.divider()
st.caption("🚀 Optimized for speed - Real-time car damage detection")
st.caption("⚡ Processing every 5th frame at 416px for maximum performance")