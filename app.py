import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io, zipfile
import tempfile
from datetime import datetime
from pathlib import Path

# Import YOLO - ultralytics handles OpenCV internally
from ultralytics import YOLO

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="Car Damage Detection", layout="wide", page_icon="🚗")
st.title("🚗 Car Damage Detection - YOLOv11")

# Constants
WEIGHTS_FILE = "epoch10.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640

# Severity thresholds
SEVERITY_T1 = 0.25  # < 25% = Light
SEVERITY_T2 = 0.60  # 25-60% = Medium, >60% = Heavy

# ==========================
# Utility Functions
# ==========================
def to_pil_rgb(arr_bgr_or_rgb):
    """Convert ultralytics result.plot() BGR array to PIL RGB."""
    if arr_bgr_or_rgb is None:
        return None
    a = arr_bgr_or_rgb
    if a.ndim == 3 and a.shape[2] == 3:
        # Ultralytics plot() returns BGR, convert to RGB
        a = a[:, :, ::-1].copy()
    return Image.fromarray(a)

def compute_severity(mask_bin: np.ndarray, xyxy: np.ndarray):
    """Calculate severity based on mask area ratio to bbox area."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_area = int(mask_bin.sum())
    ratio = float(mask_area) / float(bbox_area)
    
    if ratio < SEVERITY_T1:
        severity = "Light"
    elif ratio < SEVERITY_T2:
        severity = "Medium" 
    else:
        severity = "Heavy"
        
    return mask_area, bbox_area, ratio, severity

def bytes_from_pil(pil_img: Image.Image, fmt="JPEG"):
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

# ==========================
# Model Loading
# ==========================
@st.cache_resource(show_spinner=True)
def load_model_from_file(uploaded_file):
    """Load YOLO model from uploaded file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        model = YOLO(tmp_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource(show_spinner=True)
def load_model_from_path():
    """Load YOLO model from local path."""
    try:
        if Path(WEIGHTS_FILE).exists():
            model = YOLO(WEIGHTS_FILE)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading local model: {str(e)}")
        return None

# ==========================
# Inference Function
# ==========================
def run_inference_on_image(model, pil_img: Image.Image, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ):
    """Run inference on single image, return overlay and detection records."""
    
    # Run prediction
    results = model.predict(source=pil_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]  # Single image result
    
    # Get annotated overlay
    annotated = r.plot()
    overlay_pil = to_pil_rgb(annotated)
    
    # Extract detection data
    records = []
    names_map = r.names  # {id: name}
    boxes = getattr(r, "boxes", None)
    masks = getattr(r, "masks", None)
    
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        
        # Handle masks if available
        if masks is not None and masks.data is not None:
            m = masks.data  # (N, H, W) float 0..1
            mask_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
        else:
            mask_np = None
            
        # Process each detection
        for i in range(len(xyxy)):
            cls_id = int(cls[i])
            cls_name = names_map.get(cls_id, str(cls_id))
            conf_i = float(confs[i])
            xyxy_i = xyxy[i]
            
            # Calculate severity
            if mask_np is not None and i < mask_np.shape[0]:
                mask_area, bbox_area, ratio, severity = compute_severity(mask_np[i], xyxy_i)
            else:
                # Fallback for bbox-only models
                bbox_area = max(1, int((xyxy_i[2]-xyxy_i[0])*(xyxy_i[3]-xyxy_i[1])))
                mask_area, ratio, severity = 0, 0.0, "Light"
                
            records.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf_i,
                "x1": int(xyxy_i[0]), "y1": int(xyxy_i[1]),
                "x2": int(xyxy_i[2]), "y2": int(xyxy_i[3]),
                "mask_area": int(mask_area),
                "bbox_area": int(bbox_area), 
                "area_ratio": float(ratio),
                "severity": severity
            })
            
    return overlay_pil, records

# ==========================
# Session State
# ==========================
if "entries" not in st.session_state:
    st.session_state.entries = []  # List of {plate, files: [(name, bytes), ...]}

# ==========================
# Sidebar - Input Only
# ==========================
with st.sidebar:
    # Model loading (silent)
    model = load_model_from_path()
    
    if model is None:
        st.warning("⚠️ Model 'best.pt' not found!")
        uploaded_model = st.file_uploader("Upload YOLO model (.pt)", type=['pt'])
        
        if uploaded_model:
            model = load_model_from_file(uploaded_model)
            
    st.divider()
    
    # Input section
    st.header("📝 Add Vehicle")
    
    # Initialize input states
    if "input_plate" not in st.session_state:
        st.session_state.input_plate = ""
    if "clear_inputs" not in st.session_state:
        st.session_state.clear_inputs = False
    
    # Clear inputs after successful add
    if st.session_state.clear_inputs:
        st.session_state.input_plate = ""
        st.session_state.clear_inputs = False
        st.rerun()
    
    # Plate input with Indonesian format validation
    plate = st.text_input(
        "Plate Number", 
        value=st.session_state.input_plate,
        placeholder="B 1234 ABC",
        max_chars=11,  # Max length for Indonesian plates
        help="Format: [A-Z] [1-4 digits] [A-Z][A-Z][A-Z]"
    )
    
    files = st.file_uploader(
        "Upload Images", 
        type=["jpg","jpeg","png"], 
        accept_multiple_files=True,
        key=f"file_uploader_{len(st.session_state.entries)}"  # Force refresh
    )
    
    add_btn = st.button("➕ Add to Queue", use_container_width=True)
    
    # Handle add button
    if add_btn:
        if not plate:
            st.warning("Masukkan nomor plat terlebih dahulu")
        elif not files:
            st.warning("Upload minimal 1 gambar")
        else:
            # Store file bytes
            packed_files = [(f.name, f.read()) for f in files]
            st.session_state.entries.append({
                "plate": plate.upper().strip(), 
                "files": packed_files
            })
            st.success(f"Ditambahkan: {plate.upper()} ({len(packed_files)} gambar)")
            
            # Clear inputs
            st.session_state.clear_inputs = True
            st.rerun()
    
    st.divider()
    
    # Show queue
    if st.session_state.entries:
        st.subheader("📋 Processing Queue")
        for idx, entry in enumerate(st.session_state.entries):
            st.text(f"• {entry['plate']} — {len(entry['files'])} gambar")
        
        if st.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state.entries = []
            st.session_state.clear_inputs = True  # Also clear inputs
            st.success("Queue dikosongkan!")
            st.rerun()

# ==========================
# Main Processing Interface
# ==========================
if model is None:
    st.error("❌ Please load a model first!")
    st.stop()

if not st.session_state.entries:
    st.info("👆 Add vehicles to the queue using the sidebar, then click **Process All** below.")
    
    # Detection Settings in main area when no entries
    st.header("⚙️ Detection Settings")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
    with col2:
        img_size = st.selectbox("Image Size", [320, 640, 960, 1280], index=1)
else:
    st.header(f"🚀 Ready to Process {len(st.session_state.entries)} Vehicle(s)")
    
    # Detection Settings
    st.subheader("⚙️ Detection Settings")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
    with col2:
        img_size = st.selectbox("Image Size", [320, 640, 960, 1280], index=1)
    
    # Show summary
    total_images = sum(len(entry['files']) for entry in st.session_state.entries)
    st.metric("Total Images to Process", total_images)
    
    # Single Process All button (removed duplicate)
    process_btn = st.button("🚀 Process All", type="primary", use_container_width=True)
    
    if process_btn:
        st.header("📊 Processing Results")
        
        all_records = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Temporary directory for ZIP export
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            orig_dir = tmp_root / "original"
            seg_dir = tmp_root / "segmented"
            orig_dir.mkdir(parents=True, exist_ok=True)
            seg_dir.mkdir(parents=True, exist_ok=True)
            
            processed_count = 0
            
            # Process each vehicle
            for entry in st.session_state.entries:
                plate = entry["plate"]
                st.subheader(f"🚗 Processing: {plate}")
                
                for file_idx, (filename, file_bytes) in enumerate(entry["files"], 1):
                    processed_count += 1
                    status_text.text(f"Processing {plate} — {filename} ({processed_count}/{total_images})")
                    
                    # Load image
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    
                    # Run inference
                    overlay_pil, detections = run_inference_on_image(
                        model, pil_img, conf=conf_threshold, imgsz=img_size
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(pil_img, caption=f"Original - {filename}", use_container_width=True)
                    with col2:
                        st.image(overlay_pil or pil_img, caption=f"Detection - {filename}", use_container_width=True)
                    
                    # Save images for export
                    safe_plate = plate.replace(" ", "_")
                    orig_name = f"{safe_plate}_{file_idx:02d}_{Path(filename).name}"
                    seg_name = f"{safe_plate}_{file_idx:02d}_{Path(filename).stem}_detected.jpg"
                    
                    (orig_dir / orig_name).write_bytes(bytes_from_pil(pil_img, "JPEG"))
                    (seg_dir / seg_name).write_bytes(bytes_from_pil(overlay_pil or pil_img, "JPEG"))
                    
                    # Store detection records
                    if detections:
                        for det_idx, detection in enumerate(detections, 1):
                            record = {
                                "plate": plate,
                                "image": orig_name,
                                "detection_id": det_idx,
                                **detection
                            }
                            all_records.append(record)
                    else:
                        # No detections found
                        all_records.append({
                            "plate": plate,
                            "image": orig_name, 
                            "detection_id": 0,
                            "class_id": -1,
                            "class_name": "no_detection",
                            "confidence": 0.0,
                            "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                            "mask_area": 0,
                            "bbox_area": 0,
                            "area_ratio": 0.0,
                            "severity": "None"
                        })
                    
                    progress_bar.progress(processed_count / total_images)
                
                st.divider()
            
            # Create results DataFrame
            df = pd.DataFrame(all_records)
            csv_path = tmp_root / "detection_results.csv"
            df.to_csv(csv_path, index=False)
            
            # Display final results
            st.header("📋 Final Results Summary")
            st.success(f"✅ Processing complete! Found {len(df)} total detections.")
            
            # Summary metrics
            if len(df) > 0 and df['class_id'].iloc[0] != -1:
                damage_summary = df[df['class_id'] != -1].groupby('class_name').agg({
                    'detection_id': 'count',
                    'severity': lambda x: x.value_counts().to_dict()
                }).rename(columns={'detection_id': 'count'})
                st.dataframe(damage_summary, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)
            
            # Create ZIP download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # Add CSV
                zf.write(str(csv_path), arcname="detection_results.csv")
                
                # Add images
                for img_path in orig_dir.rglob("*"):
                    if img_path.is_file():
                        zf.write(str(img_path), arcname=f"original/{img_path.name}")
                        
                for img_path in seg_dir.rglob("*"):
                    if img_path.is_file():
                        zf.write(str(img_path), arcname=f"segmented/{img_path.name}")
            
            zip_buffer.seek(0)
            
            # Download button
            st.download_button(
                label="⬇️ Download Results (ZIP)",
                data=zip_buffer,
                file_name=f"car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
        status_text.empty()
        progress_bar.empty()

# ==========================
# Footer
# ==========================
st.divider()
st.caption("🔧 Car Damage Detection using YOLOv11 Instance Segmentation")
st.caption("⚠️ Automated severity assessment - verify with professional inspection")