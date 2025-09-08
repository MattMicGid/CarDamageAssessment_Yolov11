import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time
from datetime import datetime
from pathlib import Path
import threading
import tempfile
import io

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ==========================
# SIMPLE CONFIG
# ==========================
st.set_page_config(page_title="Car Damage Detection", layout="wide", page_icon="🚗")

# Settings
WEIGHTS_FILE = "best.pt"
FIXED_CONF = 0.10
FIXED_IOU = 0.7
FIXED_IMGSZ = 416

# Severity thresholds
SEVERITY_T1 = 0.25
SEVERITY_T2 = 0.60

# ==========================
# SESSION STATE
# ==========================
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ==========================
# MODEL LOADING
# ==========================
@st.cache_resource(show_spinner=True)
def load_model():
    if not YOLO_AVAILABLE:
        return None
    try:
        if Path(WEIGHTS_FILE).exists():
            model = YOLO(WEIGHTS_FILE)
            model.fuse()
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==========================
# INFERENCE FUNCTION
# ==========================
def run_inference_simple(model, image):
    """
    Jalankan prediksi dan kembalikan:
    - annotated_rgb: gambar anotasi (bbox + mask + label NAMA SAJA, TANPA confidence)
    - detections: list dict hasil deteksi (nama, confidence (internal), severity, timestamp)
    """
    try:
        # Run prediction
        results = model.predict(
            source=image,
            conf=FIXED_CONF,
            iou=FIXED_IOU,
            imgsz=FIXED_IMGSZ,
            verbose=False
        )

        r = results[0]
        detections = []

        # --- Render annotated image WITHOUT confidence text ---
        # Beberapa versi Ultralytics mendukung argumen conf=..., beberapa mengabaikan.
        try:
            annotated = r.plot(labels=True, conf=False, boxes=True, masks=True)  # -> BGR
        except TypeError:
            # Fallback untuk versi lama: tidak ada argumen conf, tetap akan render label nama
            annotated = r.plot(labels=True, boxes=True, masks=True)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # --- Ekstrak detections (nama + ukuran bbox untuk severity) ---
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names if hasattr(r, "names") and r.names is not None else {}

            for box, cls_id, conf in zip(boxes, classes, confs):
                cls_name = names.get(cls_id, str(cls_id))

                # Simple severity pakai luas bbox (px^2)
                bbox_area = float((box[2] - box[0]) * (box[3] - box[1]))
                if bbox_area < 5000:
                    severity = "Light"
                elif bbox_area < 15000:
                    severity = "Medium"
                else:
                    severity = "Heavy"

                detections.append({
                    "class_name": cls_name,
                    "confidence": float(conf),   # disimpan internal, TIDAK ditampilkan
                    "severity": severity,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

        return annotated_rgb, detections

    except Exception:
        # Jika gagal, kembalikan gambar input agar UI tetap jalan
        return image, []

# ==========================
# MAIN APP
# ==========================
st.title("🚗 Car Damage Detection System")

# Load model
model = load_model()
if model is None:
    st.error("❌ Model tidak tersedia! Pastikan file 'best.pt' ada.")
    st.stop()

st.success(f"✅ Model loaded: {WEIGHTS_FILE}")

# ==========================
# TABS
# ==========================
tab1, tab2, tab3 = st.tabs(["📷 Camera Capture", "📁 Upload Images", "📊 Results"])

# ==========================
# TAB 1: CAMERA CAPTURE (Simple approach)
# ==========================
with tab1:
    st.header("📷 Camera Capture")
    st.markdown("Ambil foto menggunakan kamera untuk deteksi kerusakan mobil.")

    # Camera input using streamlit's built-in camera
    camera_image = st.camera_input("Ambil foto mobil")

    col1, col2 = st.columns(2)

    if camera_image is not None:
        # Convert to PIL Image
        image = Image.open(camera_image).convert('RGB')

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        # Process button
        if st.button("🔍 Analyze Damage", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                annotated_img, detections = run_inference_simple(model, image)

                with col2:
                    st.image(annotated_img, caption="Detection Results", use_container_width=True)

                # Show results
                if detections:
                    st.success(f"✅ Found {len(detections)} damage(s)!")

                    # Add to results
                    st.session_state.detection_results.extend(detections)

                    # Display detections (TANPA persen confidence)
                    for i, det in enumerate(detections, 1):
                        severity_emoji = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
                        st.write(f"{severity_emoji} **{det['class_name']}** - {det['severity']}")
                else:
                    st.info("✅ No damage detected!")

# ==========================
# TAB 2: UPLOAD IMAGES
# ==========================
with tab2:
    st.header("📁 Upload Images")
    st.markdown("Upload gambar mobil untuk deteksi kerusakan batch.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Pilih gambar mobil",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} file(s) uploaded")

        # Process all button
        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)

            for idx, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Processing: {uploaded_file.name}")

                # Load image
                image = Image.open(uploaded_file).convert('RGB')

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Original", use_container_width=True)

                # Process
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    annotated_img, detections = run_inference_simple(model, image)

                with col2:
                    st.image(annotated_img, caption="Detection", use_container_width=True)

                # Results
                if detections:
                    st.success(f"Found {len(detections)} damage(s) in {uploaded_file.name}")

                    # Add filename to detections
                    for det in detections:
                        det['filename'] = uploaded_file.name

                    st.session_state.detection_results.extend(detections)

                    # Tampilkan TANPA persen confidence
                    for det in detections:
                        severity_emoji = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
                        st.write(f"{severity_emoji} **{det['class_name']}** - {det['severity']}")
                else:
                    st.info(f"No damage detected in {uploaded_file.name}")

                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
                st.divider()

            progress_bar.empty()
            st.success("🎉 All images processed!")

# ==========================
# TAB 3: RESULTS
# ==========================
with tab3:
    st.header("📊 Detection Results")

    if st.session_state.detection_results:
        df = pd.DataFrame(st.session_state.detection_results)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Detections", len(df))

        with col2:
            unique_classes = df['class_name'].nunique()
            st.metric("Damage Types", unique_classes)

        with col3:
            # avg_conf tetap dihitung tapi tidak perlu ditampilkan jika ingin benar2 tanpa confidence:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

        with col4:
            heavy_count = len(df[df['severity'] == 'Heavy'])
            st.metric("🔴 Heavy Damage", heavy_count)

        # Charts
        st.subheader("📈 Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Damage Types**")
            damage_counts = df['class_name'].value_counts()
            st.bar_chart(damage_counts)

        with col2:
            st.write("**Severity Distribution**")
            severity_counts = df['severity'].value_counts()
            st.bar_chart(severity_counts)

        # Data table
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)

        # Export
        col1, col2 = st.columns(2)

        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear All Results", use_container_width=True):
                st.session_state.detection_results = []
                st.success("Results cleared!")
                st.rerun()

    else:
        st.info("🔍 No detection results yet. Use Camera Capture or Upload Images to start!")

# ==========================
# SIDEBAR INFO
# ==========================
with st.sidebar:
    st.header("ℹ️ Information")

    st.markdown("""
    **How to use:**
    1. **Camera**: Take photo with built-in camera
    2. **Upload**: Process multiple images
    3. **Results**: View analysis and export data

    **Severity Levels:**
    - 🟢 Light: Minor damage
    - 🟡 Medium: Moderate damage
    - 🔴 Heavy: Severe damage
    """)

    st.markdown("---")
    st.caption("🚗 Car Damage Detection")
    st.caption("📸 No WebRTC - Simple & Reliable")

# ==========================
# FOOTER
# ==========================
st.divider()
st.caption("🛠️ Simple car damage detection without complex streaming")
st.caption("📱 Works on all devices and networks")
