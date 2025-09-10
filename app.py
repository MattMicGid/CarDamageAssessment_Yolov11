import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import zipfile
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from datetime import datetime
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Identifikasi Kerusakan Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta
DAMAGE_CLASSES = {
    0: "Dent",
    1: "Scratch", 
    2: "Crack",
    3: "Glass Shatter",
    4: "Lamp Broken",
    5: "Tire Flat"
}

DAMAGE_COLORS = {
    "Dent": (255, 0, 0),      # Merah
    "Scratch": (0, 255, 0),   # Hijau
    "Crack": (255, 255, 0),   # Kuning
    "Glass Shatter": (255, 0, 255),  # Magenta
    "Lamp Broken": (0, 255, 255),    # Cyan
    "Tire Flat": (128, 0, 128)       # Ungu
}

@st.cache_resource
def load_model():
    """Load YOLOv11 model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def get_damage_severity(damage_counts):
    """Menentukan tingkat kerusakan berdasarkan jumlah kerusakan"""
    total_damage = sum(damage_counts.values())
    
    if total_damage == 0:
        return "Baik"
    elif total_damage <= 3:
        return "Ringan"
    elif total_damage <= 7:
        return "Sedang"
    else:
        return "Berat"

def process_single_image(model, image, car_id=""):
    """Memproses satu gambar dan mendeteksi kerusakan"""
    # Konversi PIL Image ke format yang bisa diproses YOLO
    img_array = np.array(image)
    
    # Prediksi menggunakan model
    results = model(img_array)
    
    # Inisialisasi hitungan kerusakan
    damage_counts = {damage: 0 for damage in DAMAGE_CLASSES.values()}
    
    # Gambar hasil dengan overlay
    annotated_image = img_array.copy()
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for i, box in enumerate(boxes):
            # Koordinat bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Class dan confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf > 0.5:  # Threshold confidence
                damage_type = DAMAGE_CLASSES.get(cls, "Unknown")
                damage_counts[damage_type] += 1
                
                # Gambar bounding box
                color = DAMAGE_COLORS.get(damage_type, (255, 255, 255))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{damage_type}: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    severity = get_damage_severity(damage_counts)
    
    return annotated_image, damage_counts, severity

def create_damage_report(results_data):
    """Membuat laporan kerusakan dalam format DataFrame"""
    df_data = []
    
    for car_id, data in results_data.items():
        row = {"ID_Mobil": car_id}
        row.update(data["damage_counts"])
        row["Status"] = data["severity"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Reorder kolom
    damage_cols = list(DAMAGE_CLASSES.values())
    cols = ["ID_Mobil"] + damage_cols + ["Status"]
    df = df.reindex(columns=cols, fill_value=0)
    
    return df

def create_zip_results(results_data):
    """Membuat ZIP file berisi hasil deteksi"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for car_id, data in results_data.items():
            # Simpan gambar hasil
            img_pil = Image.fromarray(cv2.cvtColor(data["annotated_image"], cv2.COLOR_BGR2RGB))
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            zip_file.writestr(f"{car_id}_detected.png", img_buffer.getvalue())
            
            # Simpan detail kerusakan
            detail = {
                "car_id": car_id,
                "damage_counts": data["damage_counts"],
                "severity": data["severity"],
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            }
            zip_file.writestr(f"{car_id}_detail.json", json.dumps(detail, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("🚗 Sistem Identifikasi Kerusakan Mobil")
    st.markdown("**Menggunakan YOLOv11 untuk Deteksi Otomatis Kerusakan Kendaraan**")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'best.pt' tersedia.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan")
    mode = st.sidebar.selectbox(
        "Pilih Mode Operasi:",
        ["Upload Foto Tunggal", "Batch Processing (ZIP)", "Scan Real-time"]
    )
    
    # Info kerusakan yang dapat dideteksi
    st.sidebar.markdown("### 🔍 Jenis Kerusakan:")
    for damage in DAMAGE_CLASSES.values():
        color_hex = "#{:02x}{:02x}{:02x}".format(*DAMAGE_COLORS[damage])
        st.sidebar.markdown(f"<span style='color:{color_hex}'>●</span> {damage}", unsafe_allow_html=True)
    
    # Mode Upload Foto Tunggal
    if mode == "Upload Foto Tunggal":
        st.header("📷 Upload Foto Mobil")
        
        uploaded_file = st.file_uploader(
            "Pilih foto mobil:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload foto mobil dalam format PNG, JPG, atau JPEG"
        )
        
        car_id = st.text_input("ID/Plat Mobil (opsional):", value="MOBIL_001")
        
        if uploaded_file is not None:
            # Tampilkan gambar asli
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, caption="Foto mobil yang diupload", use_column_width=True)
            
            if st.button("🔍 Deteksi Kerusakan", type="primary"):
                with st.spinner("Memproses deteksi..."):
                    annotated_img, damage_counts, severity = process_single_image(model, image, car_id)
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(annotated_img, caption="Hasil deteksi kerusakan", use_column_width=True)
                
                # Ringkasan hasil
                st.subheader("📊 Ringkasan Kerusakan")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("Status Kerusakan", severity)
                    st.metric("Total Kerusakan", sum(damage_counts.values()))
                
                with col4:
                    for damage, count in damage_counts.items():
                        if count > 0:
                            st.metric(damage, count)
                
                # Tabel detail
                df_single = pd.DataFrame([{
                    "ID_Mobil": car_id,
                    **damage_counts,
                    "Status": severity
                }])
                
                st.subheader("📋 Detail Kerusakan")
                st.dataframe(df_single, use_container_width=True)
    
    # Mode Batch Processing
    elif mode == "Batch Processing (ZIP)":
        st.header("📦 Batch Processing - Upload ZIP")
        st.markdown("Upload file ZIP berisi foto-foto mobil untuk diproses secara batch.")
        
        uploaded_zip = st.file_uploader(
            "Upload file ZIP:",
            type=['zip'],
            help="ZIP harus berisi foto-foto mobil (PNG, JPG, JPEG)"
        )
        
        if uploaded_zip is not None:
            if st.button("🚀 Proses Batch", type="primary"):
                results_data = {}
                
                # Ekstrak dan proses ZIP
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Cari semua file gambar
                    image_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_files.append(os.path.join(root, file))
                    
                    if not image_files:
                        st.error("Tidak ditemukan file gambar dalam ZIP.")
                        return
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Proses setiap gambar
                    for i, img_path in enumerate(image_files):
                        car_id = os.path.splitext(os.path.basename(img_path))[0]
                        status_text.text(f"Memproses {car_id}... ({i+1}/{len(image_files)})")
                        
                        try:
                            image = Image.open(img_path)
                            annotated_img, damage_counts, severity = process_single_image(model, image, car_id)
                            
                            results_data[car_id] = {
                                "annotated_image": annotated_img,
                                "damage_counts": damage_counts,
                                "severity": severity,
                                "timestamp": datetime.now().isoformat()
                            }
                        except Exception as e:
                            st.warning(f"Gagal memproses {car_id}: {e}")
                        
                        progress_bar.progress((i + 1) / len(image_files))
                    
                    status_text.text("Selesai!")
                
                # Tampilkan hasil
                if results_data:
                    st.success(f"Berhasil memproses {len(results_data)} mobil!")
                    
                    # Statistik keseluruhan
                    st.subheader("📈 Statistik Keseluruhan")
                    severity_counts = {}
                    total_damages = 0
                    
                    for data in results_data.values():
                        severity = data["severity"]
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                        total_damages += sum(data["damage_counts"].values())
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Mobil", len(results_data))
                    with col2:
                        st.metric("Total Kerusakan", total_damages)
                    with col3:
                        st.metric("Mobil Baik/Ringan", severity_counts.get("Baik", 0) + severity_counts.get("Ringan", 0))
                    with col4:
                        st.metric("Mobil Sedang/Berat", severity_counts.get("Sedang", 0) + severity_counts.get("Berat", 0))
                    
                    # Tabel laporan
                    st.subheader("📋 Laporan Kerusakan")
                    df_report = create_damage_report(results_data)
                    st.dataframe(df_report, use_container_width=True)
                    
                    # Download options
                    st.subheader("💾 Download Hasil")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download CSV
                        csv_buffer = io.StringIO()
                        df_report.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "📄 Download CSV Report",
                            data=csv_buffer.getvalue(),
                            file_name=f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Download ZIP results
                        zip_results = create_zip_results(results_data)
                        st.download_button(
                            "📦 Download ZIP Hasil",
                            data=zip_results.getvalue(),
                            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    
                    # Tampilkan beberapa contoh hasil
                    st.subheader("🖼️ Contoh Hasil Deteksi")
                    sample_ids = list(results_data.keys())[:3]  # Tampilkan 3 contoh pertama
                    
                    cols = st.columns(len(sample_ids))
                    for i, car_id in enumerate(sample_ids):
                        with cols[i]:
                            st.write(f"**{car_id}**")
                            st.write(f"Status: {results_data[car_id]['severity']}")
                            st.image(results_data[car_id]["annotated_image"], use_column_width=True)
    
    # Mode Real-time (placeholder)
    elif mode == "Scan Real-time":
        st.header("📹 Scan Real-time")
        st.info("Fitur kamera real-time sedang dalam pengembangan.")
        st.markdown("""
        **Fitur yang akan tersedia:**
        - Deteksi kerusakan langsung melalui kamera
        - Preview hasil secara real-time
        - Capture dan save hasil deteksi
        """)

if __name__ == "__main__":
    main()