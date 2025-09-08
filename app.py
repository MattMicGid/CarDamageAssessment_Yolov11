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
FIXED_CONF = 0.25
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
st.caption("📱 Works on all devices and networks")            model = YOLO(WEIGHTS_FILE)
            model.fuse()
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================
# CLEAN ANNOTATION FUNCTIONS
# ==========================
def draw_clean_boxes(image, detections, show_labels=False):
    """Draw clean bounding boxes without confidence text"""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for det in detections:
        # Get color based on class name
        class_name = det['class_name'].lower()
        color = COLORS.get(class_name, COLORS['default'])
        
        # Draw bounding box
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Draw rectangle with thick border
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Optional: Draw class name (without confidence)
        if show_labels and font:
            text = det['class_name']
            # Get text size for background
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
            # Draw text
            draw.text((x1+2, y1-text_height-2), text, fill=(255, 255, 255), font=font)
    
    return image_copy

def draw_clean_segmentation(image, results, show_labels=False):
    """Draw clean segmentation masks without text"""
    if not hasattr(results, 'masks') or results.masks is None:
        # If no masks, fall back to bounding boxes
        detections = []
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            names = results.names
            
            for box, cls_id, conf in zip(boxes, classes, confs):
                cls_name = names.get(cls_id, str(cls_id))
                detections.append({
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "bbox": box.tolist()
                })
        
        return draw_clean_boxes(image, detections, show_labels)
    
    image_np = np.array(image)
    
    # Get masks
    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names
    
    # Try to load font for labels
    if show_labels:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    overlay = image_np.copy()
    
    for mask, cls_id in zip(masks, classes):
        class_name = names.get(cls_id, str(cls_id)).lower()
        color = COLORS.get(class_name, COLORS['default'])
        
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        
        # Create colored mask
        mask_indices = mask_resized > 0.5
        overlay[mask_indices] = color
        
        # Add label if requested
        if show_labels:
            # Find centroid of mask for label placement
            y_indices, x_indices = np.where(mask_indices)
            if len(y_indices) > 0:
                centroid_x = int(np.mean(x_indices))
                centroid_y = int(np.mean(y_indices))
                
                # Convert back to PIL for text drawing
                pil_overlay = Image.fromarray(overlay)
                draw = ImageDraw.Draw(pil_overlay)
                
                text = names.get(cls_id, str(cls_id))
                draw.text((centroid_x, centroid_y), text, fill=(255, 255, 255), font=font)
                overlay = np.array(pil_overlay)
    
    # Blend with original image
    alpha = 0.6
    result = cv2.addWeighted(image_np, 1-alpha, overlay, alpha, 0)
    
    return Image.fromarray(result)

def get_severity_from_area(bbox_area):
    """Determine severity based on bounding box area"""
    if bbox_area < 5000:
        return "Light"
    elif bbox_area < 15000:
        return "Medium"
    else:
        return "Heavy"

def get_severity_from_confidence(confidence):
    """Determine severity based on confidence score"""
    if confidence < SEVERITY_T1:
        return "Light"
    elif confidence < SEVERITY_T2:
        return "Medium"
    else:
        return "Heavy"

# ==========================
# INFERENCE FUNCTION (MODIFIED)
# ==========================
def run_inference_clean(model, image, annotation_type="boxes", show_labels=False):
    try:
        # Convert PIL to numpy for YOLO
        image_np = np.array(image)
        
        # Run prediction
        results = model.predict(
            source=image_np,
            conf=FIXED_CONF,
            iou=FIXED_IOU,
            imgsz=FIXED_IMGSZ,
            verbose=False
        )
        
        r = results[0]
        detections = []
        
        # Extract detections
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            
            for box, cls_id, conf in zip(boxes, classes, confs):
                cls_name = names.get(cls_id, str(cls_id))
                
                # Calculate severity based on both area and confidence
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                severity_area = get_severity_from_area(bbox_area)
                severity_conf = get_severity_from_confidence(conf)
                
                # Use the higher severity level
                if severity_area == "Heavy" or severity_conf == "Heavy":
                    severity = "Heavy"
                elif severity_area == "Medium" or severity_conf == "Medium":
                    severity = "Medium"
                else:
                    severity = "Light"
                
                detections.append({
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "severity": severity,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "bbox": box.tolist(),
                    "area": int(bbox_area)
                })
        
        # Create clean annotated image
        if annotation_type == "segmentation":
            annotated_img = draw_clean_segmentation(image, r, show_labels)
        else:
            annotated_img = draw_clean_boxes(image, detections, show_labels)
        
        return annotated_img, detections
        
    except Exception as e:
        st.error(f"Inference error: {e}")
        return image, []

# ==========================
# MAIN APP
# ==========================
st.title("🚗 Car Damage Detection System")
st.caption("Clean annotations without confidence scores - Professional look")

# Load model
model = load_model()
if model is None:
    st.error("❌ Model tidak tersedia! Pastikan file 'best.pt' ada di direktori yang sama.")
    st.info("💡 Download model YOLO dan simpan sebagai 'best.pt'")
    st.stop()

st.success(f"✅ Model loaded successfully: {WEIGHTS_FILE}")

# ==========================
# SETTINGS SIDEBAR
# ==========================
with st.sidebar:
    st.header("🎨 Annotation Settings")
    
    annotation_type = st.selectbox(
        "Annotation Type",
        ["boxes", "segmentation"],
        help="Choose between clean bounding boxes or segmentation masks"
    )
    
    show_class_labels = st.checkbox(
        "Show Class Labels", 
        value=False,
        help="Show damage type labels on the annotated image"
    )
    
    st.markdown("---")
    st.header("⚙️ Detection Settings")
    
    st.write(f"**Confidence Threshold:** {FIXED_CONF}")
    st.write(f"**IoU Threshold:** {FIXED_IOU}")
    st.write(f"**Image Size:** {FIXED_IMGSZ}")
    
    st.markdown("---")
    st.header("🎨 Color Legend")
    st.write("**Damage Type Colors:**")
    for damage_type, color in COLORS.items():
        if damage_type != 'default':
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(f"🔸 **{damage_type.title()}**: <span style='color:{color_hex}; font-size:20px'>●</span>", 
                       unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📋 Severity Levels")
    st.markdown("""
    🟢 **Light**: Small damage, low confidence
    🟡 **Medium**: Moderate damage/confidence  
    🔴 **Heavy**: Large damage, high confidence
    """)
    
    st.markdown("---")
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **📷 Camera Tab**: Take photos with device camera
    2. **📁 Upload Tab**: Process multiple images at once
    3. **📊 Results Tab**: View analysis and export data
    
    **Features:**
    - ✨ Clean annotations (no confidence text clutter)
    - 🎨 Color-coded damage types
    - 📦 Bounding boxes or segmentation masks
    - 📈 Professional results export
    """)

# ==========================
# TABS
# ==========================
tab1, tab2, tab3, tab4 = st.tabs(["📷 Camera Capture", "📁 Upload Images", "📊 Results", "🔧 About"])

# ==========================
# TAB 1: CAMERA CAPTURE
# ==========================
with tab1:
    st.header("📷 Camera Capture")
    st.markdown("Ambil foto menggunakan kamera perangkat untuk deteksi kerusakan mobil secara real-time.")
    
    # Camera input
    camera_image = st.camera_input("📸 Ambil foto mobil yang akan dianalisis")
    
    if camera_image is not None:
        col1, col2 = st.columns(2)
        
        # Convert to PIL Image
        image = Image.open(camera_image).convert('RGB')
        
        with col1:
            st.image(image, caption="🖼️ Original Image", use_container_width=True)
        
        # Analysis section
        st.markdown("### 🔍 Analysis Controls")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            analyze_btn = st.button("🚀 Analyze Damage", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("💾 Save Original", use_container_width=True):
                # Convert to bytes for download
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                st.download_button(
                    "⬇️ Download Original Image",
                    img_bytes.getvalue(),
                    f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    "image/png"
                )
        
        if analyze_btn:
            with st.spinner("🔄 Analyzing image... Please wait"):
                start_time = time.time()
                annotated_img, detections = run_inference_clean(
                    model, image, annotation_type, show_class_labels
                )
                processing_time = time.time() - start_time
                
                with col2:
                    st.image(annotated_img, 
                           caption=f"✨ Clean {annotation_type.title()} Detection", 
                           use_container_width=True)
                
                # Processing info
                st.info(f"⚡ Processing completed in {processing_time:.2f} seconds")
                
                # Results section
                st.markdown("### 📋 Detection Results")
                
                if detections:
                    st.success(f"✅ Found {len(detections)} damage(s)!")
                    
                    # Add to session results
                    for det in detections:
                        det['source'] = 'camera'
                    st.session_state.detection_results.extend(detections)
                    
                    # Display detections in a nice format
                    for i, det in enumerate(detections, 1):
                        severity_emoji = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
                        damage_type = det['class_name'].lower()
                        color = COLORS.get(damage_type, COLORS['default'])
                        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                        
                        with st.container():
                            col_icon, col_info = st.columns([1, 4])
                            with col_icon:
                                st.markdown(f"<span style='color:{color_hex}; font-size:24px'>●</span>", 
                                          unsafe_allow_html=True)
                            with col_info:
                                st.markdown(f"**{i}. {det['class_name'].title()}** {severity_emoji}")
                                st.caption(f"Severity: {det['severity']} | Confidence: {det['confidence']:.1%} | Area: {det['area']:,}px²")
                    
                    # Save annotated image option
                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        # Save annotated image
                        img_bytes = io.BytesIO()
                        annotated_img.save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        
                        st.download_button(
                            "💾 Download Annotated Image",
                            img_bytes.getvalue(),
                            f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            "image/png",
                            use_container_width=True
                        )
                    
                    with col_save2:
                        # Export detection data
                        detection_df = pd.DataFrame(detections)
                        csv_data = detection_df.to_csv(index=False)
                        st.download_button(
                            "📊 Export Detection Data",
                            csv_data,
                            f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                else:
                    st.success("✅ No damage detected! Vehicle appears to be in good condition.")
                    
                    # Still offer to save the clean image
                    img_bytes = io.BytesIO()
                    annotated_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    st.download_button(
                        "💾 Download Processed Image",
                        img_bytes.getvalue(),
                        f"no_damage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        "image/png"
                    )

# ==========================
# TAB 2: UPLOAD IMAGES
# ==========================
with tab2:
    st.header("📁 Batch Image Processing")
    st.markdown("Upload multiple images untuk analisis kerusakan secara batch processing.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📎 Pilih gambar mobil (mendukung multiple files)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Format yang didukung: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) berhasil diupload")
        
        # Show file list
        with st.expander("📋 File List"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size/1024:.1f} KB)")
        
        # Processing options
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            process_all_btn = st.button("🚀 Process All Images", type="primary", use_container_width=True)
        
        with col_opt2:
            show_progress = st.checkbox("📊 Show Progress Details", value=True)
        
        with col_opt3:
            export_after_process = st.checkbox("💾 Auto Export Results", value=False)
        
        if process_all_btn:
            # Initialize results for this batch
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                current_progress = (idx) / len(uploaded_files)
                progress_bar.progress(current_progress)
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                if show_progress:
                    st.markdown(f"### 🔄 Processing: {uploaded_file.name}")
                
                try:
                    # Load image
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    if show_progress:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original", use_container_width=True)
                    
                    # Process image
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        start_time = time.time()
                        annotated_img, detections = run_inference_clean(
                            model, image, annotation_type, show_class_labels
                        )
                        processing_time = time.time() - start_time
                    
                    if show_progress:
                        with col2:
                            st.image(annotated_img, 
                                   caption=f"Clean {annotation_type.title()}", 
                                   use_container_width=True)
                    
                    # Process results
                    if detections:
                        if show_progress:
                            st.success(f"✅ Found {len(detections)} damage(s) in {uploaded_file.name}")
                        
                        # Add metadata to detections
                        for det in detections:
                            det['filename'] = uploaded_file.name
                            det['source'] = 'upload'
                            det['processing_time'] = processing_time
                        
                        batch_results.extend(detections)
                        st.session_state.detection_results.extend(detections)
                        
                        if show_progress:
                            # Display detections
                            for det in detections:
                                severity_emoji = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
                                damage_type = det['class_name'].lower()
                                color = COLORS.get(damage_type, COLORS['default'])
                                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                                
                                st.markdown(f"<span style='color:{color_hex}'>●</span> {severity_emoji} **{det['class_name']}** - {det['severity']} ({det['confidence']:.1%})", 
                                          unsafe_allow_html=True)
                    else:
                        if show_progress:
                            st.info(f"✅ No damage detected in {uploaded_file.name}")
                    
                    if show_progress:
                        st.caption(f"⚡ Processed in {processing_time:.2f} seconds")
                        st.divider()
                
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Final progress
            progress_bar.progress(1.0)
            status_text.text("✅ All images processed!")
            
            # Batch summary
            st.markdown("## 📊 Batch Processing Summary")
            
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                st.metric("Images Processed", len(uploaded_files))
            
            with col_sum2:
                st.metric("Total Detections", len(batch_results))
            
            with col_sum3:
                if batch_results:
                    avg_conf = np.mean([det['confidence'] for det in batch_results])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("Avg Confidence", "N/A")
            
            with col_sum4:
                heavy_damage = len([det for det in batch_results if det['severity'] == 'Heavy'])
                st.metric("🔴 Heavy Damage", heavy_damage)
            
            # Export options
            if batch_results and export_after_process:
                batch_df = pd.DataFrame(batch_results)
                csv_data = batch_df.to_csv(index=False)
                
                st.download_button(
                    "📥 Download Batch Results",
                    csv_data,
                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            progress_bar.empty()
            status_text.empty()
            
            if batch_results:
                st.success(f"🎉 Batch processing completed! Found {len(batch_results)} total damage(s) across {len(uploaded_files)} image(s).")
            else:
                st.success("🎉 Batch processing completed! No damage detected in any images.")

# ==========================
# TAB 3: RESULTS
# ==========================
with tab3:
    st.header("📊 Detection Results & Analytics")
    
    if st.session_state.detection_results:
        # Prepare dataframe
        df = pd.DataFrame(st.session_state.detection_results)
        
        # Overview metrics
        st.markdown("### 📈 Overview Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Detections", len(df))
        
        with col2:
            unique_classes = df['class_name'].nunique()
            st.metric("Damage Types", unique_classes)
        
        with col3:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        with col4:
            heavy_count = len(df[df['severity'] == 'Heavy'])
            st.metric("🔴 Heavy Damage", heavy_count)
        
        with col5:
            if 'filename' in df.columns:
                unique_files = df['filename'].nunique()
                st.metric("Images Analyzed", unique_files)
            else:
                camera_count = len(df[df['source'] == 'camera']) if 'source' in df.columns else len(df)
                st.metric("Camera Shots", camera_count)
        
        # Detailed Analytics
        st.markdown("### 📊 Detailed Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 🎯 Damage Types Distribution")
            damage_counts = df['class_name'].value_counts()
            st.bar_chart(damage_counts)
            
            # Most common damage type
            most_common = damage_counts.index[0] if len(damage_counts) > 0 else "None"
            st.info(f"Most common damage: **{most_common}** ({damage_counts.iloc[0]} occurrences)")
        
        with col_chart2:
            st.markdown("#### ⚠️ Severity Distribution")
            severity_counts = df['severity'].value_counts()
            st.bar_chart(severity_counts)
            
            # Risk assessment
            risk_level = "Low"
            if heavy_count > len(df) * 0.3:
                risk_level = "High"
            elif heavy_count > 0:
                risk_level = "Medium"
            
            if risk_level == "High":
                st.error(f"🚨 Risk Level: **{risk_level}** - Immediate attention required!")
            elif risk_level == "Medium":
                st.warning(f"⚠️ Risk Level: **{risk_level}** - Schedule inspection soon")
            else:
                st.success(f"✅ Risk Level: **{risk_level}** - Vehicle in good condition")
        
        # Confidence analysis
        if len(df) > 1:
            st.markdown("#### 📈 Confidence Score Analysis")
            st.line_chart(df['confidence'])
            
            conf_stats = df['confidence'].describe()
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Min Confidence", f"{conf_stats['min']:.1%}")
            with col_stat2:
                st.metric("Max Confidence", f"{conf_stats['max']:.1%}")
            with col_stat3:
                st.metric("Std Deviation", f"{conf_stats['std']:.2%}")
        
        # Data table with filtering
        st.markdown("### 📋 Detailed Detection Data")
        
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            damage_filter = st.selectbox("Filter by Damage Type", 
                                       ["All"] + list(df['class_name'].unique()))
        
        with col_filter2:
            severity_filter = st.selectbox("Filter by Severity", 
                                         ["All", "Light", "Medium", "Heavy"])
        
        with col_filter3:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Apply filters
        filtered_df = df.copy()
        
        if damage_filter != "All":
            filtered_df = filtered_df[filtered_df['class_name'] == damage_filter]
        
        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df['severity'] == severity_filter]
        
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display filtered results
        if len(filtered_df) > 0:
            st.info(f"Showing {len(filtered_df)} of {len(df)} detections")
            
            # Remove complex columns for display
            display_columns = ['class_name', 'confidence', 'severity', 'timestamp']
            if 'filename' in filtered_df.columns:
                display_columns.append('filename')
            if 'area' in filtered_df.columns:
                display_columns.append('area')
            
            display_df = filtered_df[display_columns].copy()
            
            # Format confidence as percentage
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.warning("No results match the current filters.")
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        col_export1, col_export2, col_export3, col_export4 = st.columns(4)
        
        with col_export1:
            # Full CSV export
            export_df = df.drop(columns=['bbox'], errors='ignore')
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Results",
                csv_data,
                f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Filtered results export
            if len(filtered_df) > 0:
                filtered_export = filtered_df.drop(columns=['bbox'], errors='ignore')
                filtered_csv = filtered_export.to_csv(index=False)
                st.download_button(
                    "📊 Download Filtered",
                    filtered_csv,
                    f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.button("📊 Download Filtered", disabled=True, use_container_width=True)
        
        with col_export3:
            # Summary report
            summary_data = {
                "Total Detections": [len(df)],
                "Damage Types": [df['class_name'].nunique()],
                "Average Confidence": [f"{df['confidence'].mean():.1%}"],
                "Heavy Damage Count": [len(df[df['severity'] == 'Heavy'])],
                "Most Common Damage": [df['class_name'].value_counts().index[0] if len(df) > 0 else "None"],
                "Generated": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                "📋 Download Summary",
                summary_csv,
                f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_export4:
            # Clear results
            if st.button("🗑️ Clear All Results", use_container_width=True):
                st.session_state.detection_results = []
                st.success("✅ All results cleared!")
                st.rerun()
        
        # Advanced analytics
        if len(df) > 5:
            st.markdown("### 🔬 Advanced Analytics")
            
            # Time-based analysis if we have timestamps
            if 'timestamp' in df.columns:
                st.markdown("#### ⏰ Detection Timeline")
                # Convert timestamp to datetime for better plotting
                try:
                    df_time = df.copy()
                    df_time['hour'] = pd.to_datetime(df_time['timestamp'], format='%H:%M:%S').dt.hour
                    hourly_counts = df_time.groupby('hour').size()
                    st.bar_chart(hourly_counts)
                    
                    peak_hour = hourly_counts.idxmax()
                    st.info(f"🕐 Peak detection hour: {peak_hour}:00 ({hourly_counts.max()} detections)")
                except:
                    st.info("⏰ Timeline analysis not available")
            
            # Correlation analysis
            if 'area' in df.columns:
                st.markdown("#### 📐 Size vs Confidence Correlation")
                correlation_data = df[['area', 'confidence']].copy()
                st.scatter_chart(correlation_data.set_index('area'))
                
                correlation = df['area'].corr(df['confidence'])
                if abs(correlation) > 0.5:
                    st.info(f"📊 Strong correlation detected: {correlation:.2f}")
                else:
                    st.info(f"📊 Weak correlation: {correlation:.2f}")
    
    else:
        # No results yet
        st.info("🔍 No detection results yet. Use **Camera Capture** or **Upload Images** to start analyzing!")
        
        # Quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            #### 📷 Camera Method
            1. Go to **Camera Capture** tab
            2. Click the camera button to take a photo
            3. Click **Analyze Damage** button
            4. View clean annotations and results
            """)
        
        with col_guide2:
            st.markdown("""
            #### 📁 Upload Method
            1. Go to **Upload Images** tab
            2. Select multiple image files
            3. Click **Process All Images**
            4. Export batch results
            """)
        
        # Sample data preview
        st.markdown("### 📊 Expected Results Format")
        sample_data = {
            "class_name": ["dent", "scratch", "crack"],
            "confidence": ["85.2%", "92.1%", "76.8%"],
            "severity": ["Medium", "Heavy", "Light"],
            "timestamp": ["14:30:15", "14:30:16", "14:30:17"]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

# ==========================
# TAB 4: ABOUT
# ==========================
with tab4:
    st.header("🔧 About Car Damage Detection System")
    
    st.markdown("""
    ### 🎯 System Overview
    
    This professional car damage detection system uses advanced **YOLO (You Only Look Once)** 
    computer vision technology to automatically identify and classify vehicle damage with 
    clean, professional annotations.
    
    ### ✨ Key Features
    
    - **🎨 Clean Annotations**: No cluttered confidence text - professional look
    - **🎯 Multi-Type Detection**: Supports dent, scratch, crack, rust, and more
    - **📊 Real-time Analysis**: Fast processing with immediate results
    - **🔄 Batch Processing**: Handle multiple images simultaneously  
    - **📈 Advanced Analytics**: Comprehensive reporting and statistics
    - **💾 Export Capabilities**: CSV, image, and summary report exports
    - **📱 Mobile Friendly**: Works on all devices and screen sizes
    """)
    
    # Technical specifications
    st.markdown("### ⚙️ Technical Specifications")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        #### 🧠 AI Model Details
        - **Architecture**: YOLO (You Only Look Once)
        - **Framework**: Ultralytics YOLOv8/v11
        - **Input Size**: 416x416 pixels
        - **Confidence Threshold**: 15%
        - **IoU Threshold**: 70%
        """)
    
    with col_tech2:
        st.markdown("""
        #### 📊 Detection Capabilities
        - **Damage Types**: Dent, Scratch, Crack, Rust, Broken
        - **Severity Levels**: Light, Medium, Heavy
        - **Annotation Types**: Bounding Boxes, Segmentation
        - **Color Coding**: Automatic damage type identification
        """)
    
    # Performance metrics
    st.markdown("### 📈 Performance Metrics")
    
    # Create sample performance data
    performance_data = {
        "Metric": ["Processing Speed", "Average Confidence", "Detection Accuracy", "False Positive Rate"],
        "Value": ["< 2 seconds", "85-95%", "High", "Low"],
        "Description": [
            "Time to process single image",
            "Typical confidence range",
            "Based on model training",
            "Minimal false detections"
        ]
    }
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Usage statistics (if available)
    if st.session_state.detection_results:
        st.markdown("### 📊 Current Session Statistics")
        
        session_df = pd.DataFrame(st.session_state.detection_results)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Images Processed", 
                     session_df['filename'].nunique() if 'filename' in session_df.columns else len(session_df))
        
        with col_stat2:
            st.metric("Total Detections", len(session_df))
        
        with col_stat3:
            avg_processing_time = session_df['processing_time'].mean() if 'processing_time' in session_df.columns else 0
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s" if avg_processing_time > 0 else "N/A")
        
        with col_stat4:
            accuracy_estimate = session_df['confidence'].mean()
            st.metric("Session Accuracy", f"{accuracy_estimate:.1%}")
    
    # Color reference
    st.markdown("### 🎨 Color Reference Guide")
    
    color_guide = []
    for damage_type, color in COLORS.items():
        if damage_type != 'default':
            color_guide.append({
                "Damage Type": damage_type.title(),
                "Color (RGB)": f"({color[0]}, {color[1]}, {color[2]})",
                "Hex Code": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                "Usage": "Bounding boxes and segmentation masks"
            })
    
    color_df = pd.DataFrame(color_guide)
    st.dataframe(color_df, use_container_width=True, hide_index=True)
    
    # System requirements
    st.markdown("### 💻 System Requirements")
    
    col_req1, col_req2 = st.columns(2)
    
    with col_req1:
        st.markdown("""
        #### Minimum Requirements
        - **RAM**: 4GB+ recommended
        - **Storage**: 2GB for model files
        - **Internet**: Required for initial setup
        - **Browser**: Modern browser with camera support
        """)
    
    with col_req2:
        st.markdown("""
        #### Supported Formats
        - **Input**: JPG, JPEG, PNG, BMP, TIFF
        - **Output**: PNG (annotated images), CSV (data)
        - **Camera**: Built-in device camera support
        - **Batch**: Multiple file processing
        """)
    
    # Troubleshooting
    st.markdown("### 🔧 Troubleshooting")
    
    with st.expander("🚨 Common Issues & Solutions"):
        st.markdown("""
        **Issue**: Model not loading
        - **Solution**: Ensure 'best.pt' file exists in the same directory
        
        **Issue**: Camera not working
        - **Solution**: Allow camera permissions in browser settings
        
        **Issue**: Slow processing
        - **Solution**: Reduce image size or close other applications
        
        **Issue**: No detections found
        - **Solution**: Ensure good lighting and clear damage visibility
        
        **Issue**: Low confidence scores
        - **Solution**: Take closer photos with better angles
        """)
    
    # Version information
    st.markdown("### 📋 Version Information")
    
    version_info = {
        "Component": ["Application", "YOLO Framework", "Streamlit", "Python"],
        "Version": ["1.0.0", "Ultralytics", "Latest", "3.8+"],
        "Status": ["✅ Active", "✅ Loaded", "✅ Running", "✅ Compatible"]
    }
    version_df = pd.DataFrame(version_info)
    st.dataframe(version_df, use_container_width=True, hide_index=True)
    
    # Contact and support
    st.markdown("### 📞 Support & Contact")
    
    st.info("""
    🔧 **Technical Support**: For technical issues or questions
    📧 **Feature Requests**: Suggestions for new features
    🐛 **Bug Reports**: Report any issues encountered
    📚 **Documentation**: Additional user guides and tutorials
    """)
    
    # Credits and acknowledgments
    st.markdown("### 🙏 Credits & Acknowledgments")
    
    st.markdown("""
    - **YOLO**: Ultralytics team for the excellent YOLO framework
    - **Streamlit**: For the amazing web application framework
    - **OpenCV**: Computer vision processing capabilities
    - **PIL/Pillow**: Image processing and manipulation
    - **Pandas**: Data analysis and manipulation tools
    """)

# ==========================
# FOOTER
# ==========================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🛠️ Car Damage Detection System v1.0")

with footer_col2:
    st.caption("📸 Clean Professional Annotations")

with footer_col3:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==========================
# ADDITIONAL STYLING (Optional)
# ==========================
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #bee5eb;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)# ==========================
def run_inference_simple(model, image):
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
        
        # Get annotated image
        annotated = r.plot()  # This returns BGR
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Extract detections
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            
            for box, cls_id, conf in zip(boxes, classes, confs):
                cls_name = names.get(cls_id, str(cls_id))
                
                # Simple severity
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                if bbox_area < 5000:
                    severity = "Light"
                elif bbox_area < 15000:
                    severity = "Medium"
                else:
                    severity = "Heavy"
                
                detections.append({
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "severity": severity,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
        return annotated_rgb, detections
        
    except Exception as e:
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
                    
                    # Display detections
                    for i, det in enumerate(detections, 1):
                        severity_emoji = {"Light": "🟢", "Medium": "🟡", "Heavy": "🔴"}.get(det['severity'], "⚪")
                        st.write(f"{severity_emoji} **{det['class_name']}** - {det['severity']} ({det['confidence']:.2%})")
                        
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
