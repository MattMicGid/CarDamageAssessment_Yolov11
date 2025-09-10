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
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}

.damage-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.severity-light { border-left: 5px solid #28a745; }
.severity-medium { border-left: 5px solid #ffc107; }
.severity-heavy { border-left: 5px solid #dc3545; }

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.queue-item {
    background: #e3f2fd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #2196f3;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'car_queue' not in st.session_state:
    st.session_state.car_queue = []
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []

@st.cache_resource
def load_model():
    """Load YOLOv11 model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'best.pt' model file is in the root directory")
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

def process_image(model, image, conf_threshold=0.5):
    """Process single image for damage detection"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference
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
    
    # Create annotated image
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Color mapping for different damage types
    color_map = {
        'dent': '#FF6B6B',
        'scratch': '#4ECDC4', 
        'crack': '#45B7D1',
        'glass_damage': '#96CEB4',
        'lamp_damage': '#FFEAA7',
        'tire_damage': '#DDA0DD'
    }
    
    if results and len(results) > 0:
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    if conf >= conf_threshold:
                        # Map class to damage type (adjust based on your model classes)
                        class_names = model.names
                        damage_type = class_names.get(int(cls), 'unknown')
                        
                        # Update damage count
                        if damage_type in damage_info:
                            damage_info[damage_type] += 1
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        color = color_map.get(damage_type, '#FF0000')
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        
                        # Add label
                        label = f"{damage_type}: {conf:.2f}"
                        draw.text((x1, y1-20), label, fill=color)
    
    return annotated_img, damage_info

def create_summary_report(car_data):
    """Create summary report DataFrame"""
    df_data = []
    for car in car_data:
        row = {
            'Car_ID': car['id'],
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
        <p>Advanced AI-powered vehicle damage detection using YOLOv11</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if not model:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Settings")
        
        mode = st.selectbox(
            "Select Mode:",
            ["🏠 Single Car Analysis", "🏢 Batch Processing", "📊 View Results"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Processing Queue")
        
        if st.session_state.car_queue:
            for i, car in enumerate(st.session_state.car_queue):
                st.markdown(f"""
                <div class="queue-item">
                    <strong>{car['id']}</strong><br>
                    Plate: {car['plate']}<br>
                    Images: {len(car['images'])}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"❌ Remove {car['id']}", key=f"remove_{i}"):
                    st.session_state.car_queue.pop(i)
                    st.rerun()
        else:
            st.info("Queue is empty")
        
        if st.session_state.car_queue:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Process All"):
                    process_queue(model)
            with col2:
                if st.button("🗑️ Clear Queue"):
                    st.session_state.car_queue = []
                    st.rerun()
    
    # Main content based on mode
    if mode == "🏠 Single Car Analysis":
        single_car_mode(model)
    elif mode == "🏢 Batch Processing":
        batch_processing_mode(model)
    else:
        view_results_mode()

def single_car_mode(model):
    st.header("🏠 Single Car Analysis")
    
    tab1, tab2 = st.tabs(["📤 Upload Images", "📷 Camera Capture"])
    
    with tab1:
        st.subheader("Upload Car Images")
        
        col1, col2 = st.columns(2)
        with col1:
            car_id = st.text_input("Car ID", value=f"CAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with col2:
            license_plate = st.text_input("License Plate", value="")
        
        uploaded_files = st.file_uploader(
            "Choose car images", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files and car_id:
            images = []
            for file in uploaded_files:
                image = Image.open(file)
                images.append({'name': file.name, 'image': image})
            
            if st.button("🔍 Analyze Car"):
                analyze_single_car(model, car_id, license_plate, images)
    
    with tab2:
        st.subheader("Camera Capture")
        st.info("📱 Camera capture feature would be implemented for real-time scanning")
        st.code("""
        # Camera capture implementation would go here
        # Using streamlit-webrtc or similar library
        camera_input = st.camera_input("Take a photo")
        if camera_input:
            # Process camera image
            pass
        """)

def batch_processing_mode(model):
    st.header("🏢 Batch Processing Mode")
    
    tab1, tab2 = st.tabs(["➕ Add Cars", "📦 ZIP Upload"])
    
    with tab1:
        st.subheader("Add Individual Cars to Queue")
        
        col1, col2 = st.columns(2)
        with col1:
            car_id = st.text_input("Car ID", value=f"CAR_{len(st.session_state.car_queue)+1:03d}")
        with col2:
            license_plate = st.text_input("License Plate")
        
        uploaded_files = st.file_uploader(
            "Upload car images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if st.button("➕ Add to Queue") and uploaded_files:
            images = []
            for file in uploaded_files:
                image = Image.open(file)
                images.append({'name': file.name, 'image': image})
            
            car_data = {
                'id': car_id,
                'plate': license_plate,
                'images': images
            }
            
            st.session_state.car_queue.append(car_data)
            st.success(f"✅ Added {car_id} to queue with {len(images)} images")
            st.rerun()
    
    with tab2:
        st.subheader("Upload ZIP File")
        st.info("📦 Upload a ZIP file containing folders of car images")
        
        zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
        
        if zip_file:
            if st.button("📦 Extract and Add to Queue"):
                extract_zip_to_queue(zip_file)

def analyze_single_car(model, car_id, license_plate, images):
    """Analyze a single car"""
    st.subheader(f"🔍 Analyzing {car_id}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_damages = {
        'dent': 0, 'scratch': 0, 'crack': 0,
        'glass_damage': 0, 'lamp_damage': 0, 'tire_damage': 0
    }
    
    results_container = st.container()
    
    for i, img_data in enumerate(images):
        status_text.text(f"Processing image {i+1}/{len(images)}: {img_data['name']}")
        
        # Process image
        annotated_img, damage_info = process_image(model, img_data['image'])
        
        # Update total damages
        for damage_type, count in damage_info.items():
            total_damages[damage_type] += count
        
        # Display result
        with results_container:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_data['image'], caption=f"Original: {img_data['name']}", use_column_width=True)
            with col2:
                st.image(annotated_img, caption=f"Detected: {img_data['name']}", use_column_width=True)
            
            # Show damage info for this image
            if sum(damage_info.values()) > 0:
                st.write("**Damages detected in this image:**")
                damage_cols = st.columns(6)
                for j, (damage_type, count) in enumerate(damage_info.items()):
                    if count > 0:
                        damage_cols[j % 6].metric(damage_type.replace('_', ' ').title(), count)
        
        progress_bar.progress((i + 1) / len(images))
    
    # Final summary
    severity, severity_type = get_damage_severity(total_damages)
    
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Car ID", car_id)
        st.metric("License Plate", license_plate or "N/A")
        st.metric("Total Images", len(images))
    
    with col2:
        st.metric("Total Damages", sum(total_damages.values()))
        if severity_type == "success":
            st.success(f"Severity: {severity}")
        elif severity_type == "warning":
            st.warning(f"Severity: {severity}")
        else:
            st.error(f"Severity: {severity}")
    
    with col3:
        st.write("**Damage Breakdown:**")
        for damage_type, count in total_damages.items():
            if count > 0:
                st.write(f"• {damage_type.replace('_', ' ').title()}: {count}")
    
    status_text.text("✅ Analysis completed!")

def process_queue(model):
    """Process all cars in the queue"""
    if not st.session_state.car_queue:
        st.warning("Queue is empty!")
        return
    
    st.header("🚀 Processing Queue")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_cars = []
    
    for i, car_data in enumerate(st.session_state.car_queue):
        status_text.text(f"Processing {car_data['id']}...")
        
        total_damages = {
            'dent': 0, 'scratch': 0, 'crack': 0,
            'glass_damage': 0, 'lamp_damage': 0, 'tire_damage': 0
        }
        
        processed_images = []
        
        # Process each image for this car
        for img_data in car_data['images']:
            annotated_img, damage_info = process_image(model, img_data['image'])
            
            for damage_type, count in damage_info.items():
                total_damages[damage_type] += count
            
            processed_images.append({
                'name': img_data['name'],
                'original': img_data['image'],
                'annotated': annotated_img,
                'damages': damage_info
            })
        
        severity = get_damage_severity(total_damages)
        
        processed_car = {
            'id': car_data['id'],
            'plate': car_data['plate'],
            'images': processed_images,
            'total_damages': total_damages,
            'severity': severity
        }
        
        processed_cars.append(processed_car)
        progress_bar.progress((i + 1) / len(st.session_state.car_queue))
    
    # Save results
    st.session_state.processed_results = processed_cars
    st.session_state.car_queue = []  # Clear queue after processing
    
    status_text.text("✅ All cars processed successfully!")
    st.success(f"Processed {len(processed_cars)} cars successfully!")
    
    # Show summary
    show_batch_summary(processed_cars)

def show_batch_summary(processed_cars):
    """Show batch processing summary"""
    st.subheader("📊 Batch Processing Summary")
    
    # Create summary DataFrame
    summary_df = create_summary_report(processed_cars)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cars", len(processed_cars))
    with col2:
        st.metric("Total Images", summary_df['Total_Images'].sum())
    with col3:
        st.metric("Total Damages", summary_df['Total_Damages'].sum())
    with col4:
        heavy_damage_count = (summary_df['Severity'] == 'Heavy').sum()
        st.metric("Heavy Damage Cars", heavy_damage_count)
    
    # Display summary table
    st.dataframe(summary_df, use_container_width=True)
    
    # Download options
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Report",
            data=csv_data,
            file_name=f"car_damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("📦 Generate ZIP with Images"):
            zip_data = create_results_zip(processed_cars)
            st.download_button(
                label="📦 Download ZIP with Images",
                data=zip_data,
                file_name=f"car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

def view_results_mode():
    """View previously processed results"""
    st.header("📊 View Results")
    
    if not st.session_state.processed_results:
        st.info("No results available. Process some cars first!")
        return
    
    # Show summary
    show_batch_summary(st.session_state.processed_results)
    
    # Individual car details
    st.subheader("🔍 Individual Car Details")
    
    car_ids = [car['id'] for car in st.session_state.processed_results]
    selected_car = st.selectbox("Select car to view details:", car_ids)
    
    if selected_car:
        car_data = next(car for car in st.session_state.processed_results if car['id'] == selected_car)
        
        st.write(f"**Car ID:** {car_data['id']}")
        st.write(f"**License Plate:** {car_data['plate']}")
        st.write(f"**Severity:** {car_data['severity'][0]}")
        
        # Show images
        for img_data in car_data['images']:
            st.write(f"**Image: {img_data['name']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_data['original'], caption="Original", use_column_width=True)
            with col2:
                st.image(img_data['annotated'], caption="Detected", use_column_width=True)

def extract_zip_to_queue(zip_file):
    """Extract ZIP file and add cars to queue"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get all files in zip
            file_list = zip_ref.namelist()
            
            # Group files by folder (assuming each folder is a car)
            car_folders = {}
            for file_path in file_list:
                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    folder = os.path.dirname(file_path)
                    if folder not in car_folders:
                        car_folders[folder] = []
                    car_folders[folder].append(file_path)
            
            # Add each car folder to queue
            for folder, files in car_folders.items():
                car_id = folder.replace('/', '_') or f"ZIP_CAR_{len(st.session_state.car_queue)+1}"
                images = []
                
                for file_path in files:
                    with zip_ref.open(file_path) as img_file:
                        image = Image.open(img_file)
                        images.append({
                            'name': os.path.basename(file_path),
                            'image': image.copy()
                        })
                
                car_data = {
                    'id': car_id,
                    'plate': '',
                    'images': images
                }
                
                st.session_state.car_queue.append(car_data)
            
            st.success(f"✅ Added {len(car_folders)} cars from ZIP file to queue")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")

def create_results_zip(processed_cars):
    """Create ZIP file with all results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV report
        summary_df = create_summary_report(processed_cars)
        csv_data = summary_df.to_csv(index=False)
        zip_file.writestr("summary_report.csv", csv_data)
        
        # Add images for each car
        for car in processed_cars:
            car_folder = f"{car['id']}/"
            
            for img_data in car['images']:
                # Save original image
                original_buffer = io.BytesIO()
                img_data['original'].save(original_buffer, format='PNG')
                zip_file.writestr(f"{car_folder}original_{img_data['name']}", original_buffer.getvalue())
                
                # Save annotated image
                annotated_buffer = io.BytesIO()
                img_data['annotated'].save(annotated_buffer, format='PNG')
                zip_file.writestr(f"{car_folder}detected_{img_data['name']}", annotated_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

if __name__ == "__main__":
    main()