import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
import io, zipfile
import os
from typing import List, Dict, Any
import tempfile

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="Car Damage Detection", layout="wide", page_icon="🚗")
st.title("🚗 Car Damage Detection App")

# Model weights path
WEIGHTS_FILE = "best.pt"

# Damage severity mapping
DAMAGE_SEVERITY = {
    "scratch": 1,       # cosmetic
    "dent": 2,          # bodywork needed  
    "crack": 3,         # structural concern
    "glass_shatter": 3, # safety issue
    "lamp_broken": 2,   # functional issue
    "tire_flat": 2      # operational issue
}

# ==========================
# Model Loading Functions
# ==========================
@st.cache_resource(show_spinner=True)
def load_model_from_file(uploaded_file):
    """Load YOLO model from uploaded file."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        model = YOLO(tmp_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource(show_spinner=True)
def load_model_from_path():
    """Load YOLO model from local path."""
    try:
        if os.path.exists(WEIGHTS_FILE):
            model = YOLO(WEIGHTS_FILE)
            st.success("✅ Model loaded from local file!")
            return model
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading local model: {str(e)}")
        return None

# ==========================
# Severity Calculation
# ==========================
def calculate_severity(damage_type: str, confidence: float, bbox_area_ratio: float) -> str:
    """
    Calculate damage severity based on:
    1. Damage Type Priority (60%)
    2. Detection Confidence (25%) 
    3. Size Factor (15%)
    """
    # Get damage type score
    type_score = DAMAGE_SEVERITY.get(damage_type.lower().replace(' ', '_'), 1)
    
    # Size factor (normalize bbox area)
    size_factor = min(bbox_area_ratio * 10, 3)  # cap at 3
    
    # Calculate final score
    final_score = (type_score * 0.6) + (confidence * 0.25 * 3) + (size_factor * 0.15)
    
    # Classify severity
    if final_score < 1.5:
        return "Minor"
    elif final_score < 2.5:
        return "Moderate"
    else:
        return "Severe"

# ==========================
# Detection Function
# ==========================
def run_detection(model: YOLO, images: List[Image.Image], conf_threshold: float):
    """Run YOLO detection on multiple images."""
    all_results = []
    annotated_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image in enumerate(images):
        status_text.text(f"Processing image {i+1}/{len(images)}...")
        progress_bar.progress((i + 1) / len(images))
        
        # Convert to RGB array
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]
        
        # Run inference
        results = model.predict(img_array, conf=conf_threshold, verbose=False)[0]
        
        # Get annotated image
        annotated = results.plot()
        annotated = annotated[:, :, ::-1]  # BGR to RGB
        annotated_images.append(Image.fromarray(annotated))
        
        # Extract detections
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_ratio = bbox_area / (w * h)
                
                damage_type = results.names.get(cls_id, f"class_{cls_id}")
                severity = calculate_severity(damage_type, conf, bbox_area_ratio)
                
                all_results.append({
                    'image_idx': i + 1,
                    'damage_type': damage_type,
                    'confidence': round(float(conf), 3),
                    'severity': severity,
                    'bbox_area_ratio': round(bbox_area_ratio, 4)
                })
    
    status_text.empty()
    progress_bar.empty()
    
    return all_results, annotated_images

# ==========================
# Sidebar - Model Loading
# ==========================
with st.sidebar:
    st.header("🔧 Setup")
    
    # Try to load local model first
    model = load_model_from_path()
    
    # If no local model, allow upload
    if model is None:
        st.warning("⚠️ Model file 'best.pt' not found!")
        uploaded_model = st.file_uploader(
            "Upload YOLO model (.pt file)", 
            type=['pt'], 
            help="Upload your trained YOLOv11 model file"
        )
        
        if uploaded_model:
            model = load_model_from_file(uploaded_model)
    
    st.divider()
    
    # Detection settings
    st.subheader("⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25, 
        step=0.05,
        help="Higher values = more strict detection"
    )

# ==========================
# Main App Interface
# ==========================
if model is None:
    st.error("❌ Please load a model first!")
    st.stop()

# Input section
st.header("📝 Input")
col1, col2 = st.columns([1, 3])

with col1:
    plate_number = st.text_input(
        "Plate Number (Optional)", 
        placeholder="e.g., B 1234 ABC",
        help="For identification purposes"
    )

with col2:
    uploaded_images = st.file_uploader(
        "Upload Car Images", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True,
        help="Upload multiple images of the car damage"
    )

# Process button
if uploaded_images:
    if st.button("🔍 Analyze Damage", type="primary", use_container_width=True):
        # Convert uploaded files to PIL images
        images = [Image.open(img) for img in uploaded_images]
        
        # Run detection
        detections, annotated_images = run_detection(model, images, conf_threshold)
        
        # Store results in session state
        st.session_state['detections'] = detections
        st.session_state['annotated_images'] = annotated_images
        st.session_state['original_images'] = images
        st.session_state['plate_number'] = plate_number or "Unknown"

# ==========================
# Results Display
# ==========================
if 'detections' in st.session_state:
    detections = st.session_state['detections']
    annotated_images = st.session_state['annotated_images']
    original_images = st.session_state['original_images']
    plate = st.session_state['plate_number']
    
    st.header("📊 Analysis Results")
    
    # Summary Card
    if detections:
        df = pd.DataFrame(detections)
        total_damage = len(detections)
        severity_counts = df['severity'].value_counts()
        highest_severity = df.loc[df['severity'].map({'Minor': 1, 'Moderate': 2, 'Severe': 3}).idxmax(), 'severity']
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚗 Vehicle", plate)
        with col2:
            st.metric("📷 Images", len(original_images))
        with col3:
            st.metric("⚠️ Damage Found", total_damage)
        with col4:
            severity_color = {"Minor": "🟢", "Moderate": "🟡", "Severe": "🔴"}
            st.metric("📈 Max Severity", f"{severity_color.get(highest_severity, '⚪')} {highest_severity}")
        
        st.divider()
        
        # Damage Breakdown
        st.subheader("🔍 Damage Breakdown")
        
        # Create summary table
        summary_data = []
        for damage_type in df['damage_type'].unique():
            type_df = df[df['damage_type'] == damage_type]
            severity_dist = type_df['severity'].value_counts()
            
            summary_data.append({
                'Damage Type': damage_type.title(),
                'Count': len(type_df),
                'Minor': severity_dist.get('Minor', 0),
                'Moderate': severity_dist.get('Moderate', 0),
                'Severe': severity_dist.get('Severe', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
    else:
        st.success("✅ No damage detected in the uploaded images!")
    
    st.divider()
    
    # Image Gallery
    st.subheader("🖼️ Image Gallery")
    
    for i, (original, annotated) in enumerate(zip(original_images, annotated_images)):
        st.markdown(f"**Image {i+1}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption=f"Original - Image {i+1}", use_container_width=True)
        with col2:
            st.image(annotated, caption=f"Detection Results - Image {i+1}", use_container_width=True)
        
        # Show detections for this image
        image_detections = [d for d in detections if d['image_idx'] == i+1]
        if image_detections:
            image_df = pd.DataFrame(image_detections)
            st.dataframe(image_df[['damage_type', 'confidence', 'severity']], use_container_width=True)
        else:
            st.info("No damage detected in this image")
        
        st.divider()
    
    # ==========================
    # Export Options
    # ==========================
    st.header("⬇️ Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if detections:
            # CSV Export
            export_df = pd.DataFrame(detections)
            export_df.insert(0, 'plate_number', plate)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"damage_report_{plate.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # ZIP Export (annotated images)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(annotated_images):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=90)
                zip_file.writestr(f"annotated_image_{i+1:02d}.jpg", img_buffer.getvalue())
        
        st.download_button(
            label="🖼️ Download Annotated Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"annotated_images_{plate.replace(' ', '_')}.zip",
            mime="application/zip",
            use_container_width=True
        )

# ==========================
# Footer
# ==========================
st.divider()
st.caption("⚠️ **Disclaimer:** Severity assessment is automated and should be validated by professional inspection.")
st.caption("🔧 **Model Info:** YOLOv11 Instance Segmentation for Car Damage Detection")