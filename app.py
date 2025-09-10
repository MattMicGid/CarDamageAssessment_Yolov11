import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import zipfile
import io
import os
import tempfile
from datetime import datetime
import base64

# Configure page
st.set_page_config(
    page_title="🚗 Car Damage Assessment System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.step-card {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.active-step {
    border-color: #667eea;
    background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%);
}

.queue-item {
    background: #e8f5e8;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #28a745;
}

.processing-item {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
}

.completed-item {
    background: #d1ecf1;
    border-left: 4px solid #17a2b8;
}

.damage-mask {
    border-radius: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.severity-light { 
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    padding: 0.5rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
}

.severity-medium { 
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    color: #856404;
    padding: 0.5rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
}

.severity-heavy { 
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    padding: 0.5rem;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'dashboard'
if 'car_queue' not in st.session_state:
    st.session_state.car_queue = []
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'current_license_plate' not in st.session_state:
    st.session_state.current_license_plate = ""
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

@st.cache_resource
def load_model():
    """Load YOLOv11 segmentation model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("📁 Make sure 'best.pt' model file is in the root directory")
        return None

def get_damage_severity(damage_counts):
    """Calculate damage severity based on damage counts"""
    total_damages = sum(damage_counts.values())
    
    if total_damages == 0:
        return "No Damage", "success"
    elif total_damages <= 2:
        return "Light", "success"
    elif total_damages <= 5:
        return "Medium", "warning"
    else:
        return "Heavy", "error"

def process_image_segmentation(model, image, conf_threshold=0.1):
    """Process single image for damage detection with instance segmentation"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference with segmentation
    results = model(img_array, conf=conf_threshold)
    
    # Extract damage information
    damage_info = {
        'dent': 0,
        'scratch': 0,
        'crack': 0,
        'glass_damage': 0,
        'lamp_damage': 0,
        'tire_damage': 0
    }
    
    # Create annotated image with masks
    annotated_img = image.copy()
    img_array_draw = np.array(annotated_img)
    
    # Color mapping for different damage types (RGB format)
    color_map = {
        'dent': (255, 107, 107),      # Red
        'scratch': (78, 205, 196),     # Teal  
        'crack': (69, 183, 209),       # Blue
        'glass_damage': (150, 206, 180), # Green
        'lamp_damage': (255, 234, 167),  # Yellow
        'tire_damage': (221, 160, 221)   # Purple
    }
    
    # Transparency for masks
    alpha = 0.4
    
    if results and len(results) > 0:
        for result in results:
            if result.boxes is not None and result.masks is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                masks = result.masks.data.cpu().numpy()
                
                for box, cls, conf, mask in zip(boxes, classes, confidences, masks):
                    if conf >= conf_threshold:
                        # Map class to damage type (adjust based on your model classes)
                        class_names = model.names
                        damage_type = class_names.get(int(cls), 'unknown')
                        
                        # Update damage count
                        if damage_type in damage_info:
                            damage_info[damage_type] += 1
                        
                        # Get color for this damage type
                        color = color_map.get(damage_type, (255, 0, 0))
                        
                        # Apply mask overlay
                        mask_resized = cv2.resize(mask, (img_array_draw.shape[1], img_array_draw.shape[0]))
                        mask_binary = mask_resized > 0.5
                        
                        # Create colored mask
                        colored_mask = np.zeros_like(img_array_draw)
                        colored_mask[mask_binary] = color
                        
                        # Blend with original image
                        img_array_draw = cv2.addWeighted(
                            img_array_draw, 1-alpha,
                            colored_mask, alpha,
                            0
                        )
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_array_draw, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{damage_type}: {conf:.2f}"
                        cv2.putText(img_array_draw, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Convert back to PIL
    annotated_img = Image.fromarray(img_array_draw)
    
    return annotated_img, damage_info

def create_summary_report(car_data):
    """Create summary report DataFrame"""
    df_data = []
    for car in car_data:
        row = {
            'License_Plate': car['plate'],
            'Total_Images': len(car['images']),
            'Dent': car['total_damages']['dent'],
            'Scratch': car['total_damages']['scratch'],
            'Crack': car['total_damages']['crack'],
            'Glass_Damage': car['total_damages']['glass_damage'],
            'Lamp_Damage': car['total_damages']['lamp_damage'],
            'Tire_Damage': car['total_damages']['tire_damage'],
            'Total_Damages': sum(car['total_damages'].values()),
            'Severity': car['severity'][0],
            'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df_data.append(row)
    
    return pd.DataFrame(df_data)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Car Damage Assessment System</h1>
        <p>AI-powered vehicle damage detection using YOLOv11 Instance Segmentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if not model:
        st.stop()
    
    # Flow control based on current step
    if st.session_state.current_step == 'dashboard':
        show_dashboard()
    elif st.session_state.current_step == 'input':
        show_input_step(model)
    elif st.session_state.current_step == 'queue':
        show_queue_step()
    elif st.session_state.current_step == 'process':
        show_process_step(model)
    elif st.session_state.current_step == 'results':
        show_results_step()
    elif st.session_state.current_step == 'download':
        show_download_step()

def show_dashboard():
    """Dashboard - Starting point"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>📊 Halaman Dashboard</h2>
        <p>Selamat datang di sistem identifikasi kerusakan mobil menggunakan AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚗 Cars in Queue", len(st.session_state.car_queue))
    with col2:
        total_processed = len(st.session_state.processed_results)
        st.metric("✅ Cars Processed", total_processed)
    with col3:
        if st.session_state.processed_results:
            total_damages = sum(sum(car['total_damages'].values()) for car in st.session_state.processed_results)
            st.metric("🔍 Total Damages Found", total_damages)
        else:
            st.metric("🔍 Total Damages Found", 0)
    with col4:
        if st.session_state.processed_results:
            heavy_damage = sum(1 for car in st.session_state.processed_results if car['severity'][0] == 'Heavy')
            st.metric("⚠️ Heavy Damage Cars", heavy_damage)
        else:
            st.metric("⚠️ Heavy Damage Cars", 0)
    
    st.markdown("---")
    
    # Recent activity
    if st.session_state.processed_results:
        st.subheader("📈 Recent Processing Results")
        for car in st.session_state.processed_results[-3:]:  # Show last 3
            severity_class = f"severity-{car['severity'][0].lower()}"
            st.markdown(f"""
            <div class="completed-item {severity_class}">
                <strong>🚗 {car['plate'] or 'Unknown Plate'}</strong><br>
                Severity: {car['severity'][0]} | Total Damages: {sum(car['total_damages'].values())}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Mulai Identifikasi Kerusakan", use_container_width=True, type="primary"):
            st.session_state.current_step = 'input'
            st.rerun()

def show_input_step(model):
    """Input Nomor Plat & Gambar"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>📝 Input Nomor Plat & Gambar</h2>
        <p>Masukkan nomor plat dan upload gambar mobil untuk dianalisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.current_step = 'dashboard'
            st.rerun()
    
    # Input form
    st.subheader("🚗 Detail Kendaraan")
    
    license_plate = st.text_input(
        "Nomor Plat Kendaraan", 
        value=st.session_state.current_license_plate,
        placeholder="Contoh: B 1234 CD"
    )
    st.session_state.current_license_plate = license_plate
    
    st.subheader("📸 Upload Gambar")
    uploaded_files = st.file_uploader(
        "Pilih gambar kendaraan (bisa multiple)", 
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload beberapa foto dari berbagai sudut untuk hasil yang lebih akurat"
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = []
        
        # Preview images
        st.subheader("👀 Preview Gambar")
        cols = st.columns(min(len(uploaded_files), 3))
        
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            st.session_state.uploaded_images.append({
                'name': file.name,
                'image': image
            })
            
            with cols[i % 3]:
                st.image(image, caption=file.name, use_column_width=True)
        
        st.success(f"✅ {len(uploaded_files)} gambar telah diupload")
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col2:
        can_proceed = license_plate.strip() and st.session_state.uploaded_images
        if st.button("➡️ Tambah ke Queue", disabled=not can_proceed, use_container_width=True):
            # Add to queue
            car_data = {
                'plate': license_plate.strip(),
                'images': st.session_state.uploaded_images.copy(),
                'timestamp': datetime.now()
            }
            
            st.session_state.car_queue.append(car_data)
            
            # Reset inputs
            st.session_state.current_license_plate = ""
            st.session_state.uploaded_images = []
            
            # Move to queue step
            st.session_state.current_step = 'queue'
            st.rerun()

def show_queue_step():
    """Tampilkan Queue dan opsi untuk proses"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>📋 Tambah Queue</h2>
        <p>Kelola antrian kendaraan yang akan diproses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅️ Back to Input"):
            st.session_state.current_step = 'input'
            st.rerun()
    
    # Show queue
    if st.session_state.car_queue:
        st.subheader(f"🚗 Antrian Kendaraan ({len(st.session_state.car_queue)} mobil)")
        
        for i, car in enumerate(st.session_state.car_queue):
            st.markdown(f"""
            <div class="queue-item">
                <strong>🚗 {car['plate']}</strong><br>
                📸 {len(car['images'])} gambar | 
                🕐 {car['timestamp'].strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button(f"❌ Remove", key=f"remove_{i}"):
                    st.session_state.car_queue.pop(i)
                    st.rerun()
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Tambah Mobil Lain", use_container_width=True):
                st.session_state.current_step = 'input'
                st.rerun()
        
        with col2:
            if st.button("🚀 Proses Semua", use_container_width=True, type="primary"):
                st.session_state.current_step = 'process'
                st.rerun()
        
        with col3:
            if st.button("🗑️ Kosongkan Queue", use_container_width=True):
                st.session_state.car_queue = []
                st.rerun()
    
    else:
        st.info("📭 Queue kosong. Tambahkan mobil terlebih dahulu.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("➕ Tambah Mobil Pertama", use_container_width=True):
                st.session_state.current_step = 'input'
                st.rerun()

def show_process_step(model):
    """Process all cars in queue"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>⚙️ Proses Semua</h2>
        <p>Sedang memproses semua kendaraan dalam antrian...</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.car_queue:
        st.error("❌ Queue kosong!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'
            st.rerun()
        return
    
    # Processing
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Status Processing")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    processed_cars = []
    total_cars = len(st.session_state.car_queue)
    
    for i, car_data in enumerate(st.session_state.car_queue):
        status_text.text(f"🔍 Processing {car_data['plate']} ({i+1}/{total_cars})...")
        
        # Initialize damage tracking
        total_damages = {
            'dent': 0, 'scratch': 0, 'crack': 0,
            'glass_damage': 0, 'lamp_damage': 0, 'tire_damage': 0
        }
        
        processed_images = []
        
        # Process each image
        for j, img_data in enumerate(car_data['images']):
            sub_status = f"🔍 Processing {car_data['plate']} - Image {j+1}/{len(car_data['images'])}"
            status_text.text(sub_status)
            
            # Run segmentation
            annotated_img, damage_info = process_image_segmentation(model, img_data['image'])
            
            # Update total damages
            for damage_type, count in damage_info.items():
                total_damages[damage_type] += count
            
            processed_images.append({
                'name': img_data['name'],
                'original': img_data['image'],
                'annotated': annotated_img,
                'damages': damage_info
            })
        
        # Calculate severity
        severity = get_damage_severity(total_damages)
        
        processed_car = {
            'plate': car_data['plate'],
            'images': processed_images,
            'total_damages': total_damages,
            'severity': severity,
            'processed_time': datetime.now()
        }
        
        processed_cars.append(processed_car)
        
        # Update progress
        progress_bar.progress((i + 1) / total_cars)
        
        # Show intermediate results
        with results_container:
            severity_class = f"severity-{severity[0].lower()}"
            st.markdown(f"""
            <div class="processing-item {severity_class}">
                <strong>✅ {car_data['plate']}</strong> - {severity[0]} damage<br>
                Total kerusakan ditemukan: {sum(total_damages.values())}
            </div>
            """, unsafe_allow_html=True)
    
    # Save results and clear queue
    st.session_state.processed_results = processed_cars
    st.session_state.car_queue = []
    
    status_text.text("✅ Semua kendaraan berhasil diproses!")
    
    st.success(f"🎉 Berhasil memproses {total_cars} kendaraan!")
    
    # Auto-proceed to results after 2 seconds
    import time
    time.sleep(2)
    st.session_state.current_step = 'results'
    st.rerun()

def show_results_step():
    """Display processing results with masks side by side"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>📊 Tampilkan Mask Side by Side</h2>
        <p>Hasil analisis kerusakan dengan segmentasi mask</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.processed_results:
        st.error("❌ Tidak ada hasil untuk ditampilkan!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'
            st.rerun()
        return
    
    # Summary metrics
    st.subheader("📈 Ringkasan Hasil")
    
    total_cars = len(st.session_state.processed_results)
    total_damages = sum(sum(car['total_damages'].values()) for car in st.session_state.processed_results)
    heavy_damage_cars = sum(1 for car in st.session_state.processed_results if car['severity'][0] == 'Heavy')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚗 Total Mobil", total_cars)
    col2.metric("🔍 Total Kerusakan", total_damages)
    col3.metric("⚠️ Kerusakan Berat", heavy_damage_cars)
    col4.metric("✅ Success Rate", "100%")
    
    st.markdown("---")
    
    # Display results for each car
    for car_idx, car in enumerate(st.session_state.processed_results):
        st.subheader(f"🚗 {car['plate']} - {car['severity'][0]} Damage")
        
        # Damage summary for this car
        damages = car['total_damages']
        damage_text = []
        for damage_type, count in damages.items():
            if count > 0:
                damage_text.append(f"{damage_type.replace('_', ' ').title()}: {count}")
        
        if damage_text:
            st.write(f"**Kerusakan ditemukan:** {', '.join(damage_text)}")
        else:
            st.write("**Tidak ada kerusakan ditemukan**")
        
        # Display images side by side
        for img_idx, img_data in enumerate(car['images']):
            st.write(f"**📸 {img_data['name']}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_data['original'], 
                        caption="Gambar Asli", 
                        use_column_width=True)
            
            with col2:
                st.image(img_data['annotated'], 
                        caption="Hasil Deteksi dengan Mask", 
                        use_column_width=True)
            
            # Show damage details for this image
            if sum(img_data['damages'].values()) > 0:
                st.write("**Kerusakan pada gambar ini:**")
                damage_cols = st.columns(6)
                for j, (damage_type, count) in enumerate(img_data['damages'].items()):
                    if count > 0:
                        damage_cols[j % 6].metric(
                            damage_type.replace('_', ' ').title(), 
                            count
                        )
        
        st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Prediksi Lagi", use_container_width=True):
            # Clear results and go back to dashboard
            st.session_state.processed_results = []
            st.session_state.current_step = 'dashboard'
            st.rerun()
    
    with col2:
        if st.button("📥 Download ZIP?", use_container_width=True, type="primary"):
            st.session_state.current_step = 'download'
            st.rerun()
    
    with col3:
        if st.button("📊 Tampilkan Output", use_container_width=True):
            show_detailed_output()

def show_download_step():
    """Download step - File Download"""
    st.markdown("""
    <div class="step-card active-step">
        <h2>📥 File Download</h2>
        <p>Download hasil analisis dalam berbagai format</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.processed_results:
        st.error("❌ Tidak ada hasil untuk didownload!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'
            st.rerun()
        return
    
    st.subheader("📊 Summary Report")
    
    # Create and display summary DataFrame
    summary_df = create_summary_report(st.session_state.processed_results)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Report",
            data=csv_data,
            file_name=f"car_damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # ZIP Download
        zip_data = create_results_zip(st.session_state.processed_results)
        st.download_button(
            label="📦 Download ZIP with Images",
            data=zip_data,
            file_name=f"car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Kembali ke Hasil", use_container_width=True):
            st.session_state.current_step = 'results'
            st.rerun()
    
    with col2:
        if st.button("🏠 Kembali ke Dashboard", use_container_width=True):
            st.session_state.current_step = 'dashboard'
            st.rerun()

def show_detailed_output():
    """Show detailed tabular output"""
    st.subheader("📋 Detailed Output Table")
    
    # Detailed breakdown per car per image
    detailed_data = []
    
    for car in st.session_state.processed_results:
        for img in car['images']:
            row = {
                'License_Plate': car['plate'],
                'Image_Name': img['name'],
                'Dent': img['damages']['dent'],
                'Scratch': img['damages']['scratch'],
                'Crack': img['damages']['crack'],
                'Glass_Damage': img['damages']['glass_damage'],
                'Lamp_Damage': img['damages']['lamp_damage'],
                'Tire_Damage': img['damages']['tire_damage'],
                'Total_in_Image': sum(img['damages'].values()),
                'Car_Severity': car['severity'][0]
            }
            detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(detailed_df, use_container_width=True)

def create_results_zip(processed_cars):
    """Create ZIP file with all results including masks"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV report
        summary_df = create_summary_report(processed_cars)
        csv_data = summary_df.to_csv(index=False)
        zip_file.writestr("summary_report.csv", csv_data)
        
        # Add images for each car
        for car in processed_cars:
            car_folder = f"{car['plate'].replace(' ', '_')}/"
            
            for img_data in car['images']:
                # Save original image
                original_buffer = io.BytesIO()
                img_data['original'].save(original_buffer, format='PNG')
                zip_file.writestr(f"{car_folder}original_{img_data['name']}", original_buffer.getvalue())
                
                # Save segmented image with masks
                segmented_buffer = io.BytesIO()
                img_data['annotated'].save(segmented_buffer, format='PNG')
                zip_file.writestr(f"{car_folder}segmented_{img_data['name']}", segmented_buffer.getvalue())
        
        # Add detailed report per image
        detailed_data = []
        for car in processed_cars:
            for img in car['images']:
                row = {
                    'License_Plate': car['plate'],
                    'Image_Name': img['name'],
                    'Dent': img['damages']['dent'],
                    'Scratch': img['damages']['scratch'],
                    'Crack': img['damages']['crack'],
                    'Glass_Damage': img['damages']['glass_damage'],
                    'Lamp_Damage': img['damages']['lamp_damage'],
                    'Tire_Damage': img['damages']['tire_damage'],
                    'Total_in_Image': sum(img['damages'].values()),
                    'Car_Severity': car['severity'][0],
                    'Processing_Time': car['processed_time'].strftime('%Y-%m-%d %H:%M:%S')
                }
                detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv = detailed_df.to_csv(index=False)
        zip_file.writestr("detailed_per_image_report.csv", detailed_csv)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

if __name__ == "__main__":
    main()