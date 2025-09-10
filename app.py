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
    page_title="Sistem Identifikasi Kerusakan Mobil - YOLOv11 Instance Segmentation",
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
    "Dent": (255, 100, 100),      # Merah terang
    "Scratch": (100, 255, 100),   # Hijau terang
    "Crack": (255, 255, 100),     # Kuning terang
    "Glass Shatter": (255, 100, 255),  # Magenta terang
    "Lamp Broken": (100, 255, 255),    # Cyan terang
    "Tire Flat": (200, 100, 255)       # Ungu terang
}

@st.cache_resource
def load_model():
    """Load YOLOv11 segmentation model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.error("Pastikan file 'best.pt' (YOLOv11 segmentation model) tersedia di direktori aplikasi.")
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

def apply_mask_overlay(image, mask, color, alpha=0.4):
    """Menerapkan mask overlay pada gambar dengan transparansi"""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def process_single_image(model, image, car_id=""):
    """Memproses satu gambar dengan instance segmentation"""
    # Konversi PIL Image ke format yang bisa diproses YOLO
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Prediksi menggunakan model
    results = model(img_array)
    
    # Inisialisasi hitungan kerusakan
    damage_counts = {damage: 0 for damage in DAMAGE_CLASSES.values()}
    
    # Gambar hasil dengan mask overlay
    annotated_image = img_array.copy()
    
    if len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks
        boxes = results[0].boxes
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Class
            cls = int(box.cls[0])
            damage_type = DAMAGE_CLASSES.get(cls, "Unknown")
            
            if damage_type != "Unknown":
                damage_counts[damage_type] += 1
                
                # Dapatkan mask sebagai array
                mask_array = mask.data[0].cpu().numpy()
                
                # Resize mask ke ukuran gambar asli
                mask_resized = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Terapkan mask overlay
                color = DAMAGE_COLORS.get(damage_type, (255, 255, 255))
                annotated_image = apply_mask_overlay(annotated_image, mask_binary, color, alpha=0.3)
                
                # Tambahkan label pada centroid mask
                moments = cv2.moments(mask_binary)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Label tanpa confidence
                    label = damage_type
                    
                    # Background untuk text
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(annotated_image, 
                                (cx - text_width//2 - 5, cy - text_height - 5),
                                (cx + text_width//2 + 5, cy + baseline + 5),
                                color, -1)
                    
                    # Text
                    cv2.putText(annotated_image, label, 
                              (cx - text_width//2, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    severity = get_damage_severity(damage_counts)
    
    # Konversi kembali ke RGB untuk display
    if len(annotated_image.shape) == 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
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

def process_batch_images(model, uploaded_files, progress_container):
    """Memproses multiple image files"""
    results_data = {}
    
    if not uploaded_files:
        return results_data
    
    # Progress bar
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        car_id = os.path.splitext(uploaded_file.name)[0]
        status_text.text(f"Memproses {car_id}... ({i+1}/{len(uploaded_files)})")
        
        try:
            image = Image.open(uploaded_file)
            annotated_img, damage_counts, severity = process_single_image(model, image, car_id)
            
            results_data[car_id] = {
                "original_image": np.array(image),
                "annotated_image": annotated_img,
                "damage_counts": damage_counts,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            st.warning(f"Gagal memproses {car_id}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Selesai memproses semua gambar!")
    return results_data

def main():
    st.title("🚗 Sistem Identifikasi Kerusakan Mobil")
    st.markdown("**YOLOv11 Instance Segmentation untuk Deteksi Kerusakan Bodi Kendaraan**")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Konfigurasi")
    st.sidebar.markdown("""
    **Metode:** YOLOv11 Instance Segmentation  
    **Dataset:** Car Damage Dataset (CarDD)  
    **Anotasi:** Polygon-based annotation
    """)
    
    # Info kerusakan yang dapat dideteksi
    st.sidebar.markdown("### 🎯 Kategori Kerusakan:")
    for damage in DAMAGE_CLASSES.values():
        color_rgb = DAMAGE_COLORS[damage]
        color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
        st.sidebar.markdown(f"<span style='color:{color_hex}; font-weight:bold'>●</span> {damage}", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📊 Tingkat Keparahan:")
    st.sidebar.markdown("- **Baik**: 0 kerusakan")
    st.sidebar.markdown("- **Ringan**: 1-3 kerusakan") 
    st.sidebar.markdown("- **Sedang**: 4-7 kerusakan")
    st.sidebar.markdown("- **Berat**: 8+ kerusakan")
    
    # Mode selection
    mode = st.selectbox(
        "**Pilih Mode Operasi:**",
        ["📷 Upload Foto Tunggal", "📁 Batch Processing - Multiple Files"],
        help="Pilih mode sesuai kebutuhan Anda"
    )
    
    # Mode Upload Foto Tunggal
    if mode == "📷 Upload Foto Tunggal":
        st.header("📷 Analisis Foto Tunggal")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Pilih foto mobil:",
                type=['png', 'jpg', 'jpeg'],
                help="Upload foto mobil dalam format PNG, JPG, atau JPEG"
            )
            
            car_id = st.text_input("ID/Plat Mobil:", value="", placeholder="Contoh: B1234CD")
            
            if not car_id:
                car_id = f"MOBIL_{datetime.now().strftime('%H%M%S')}"
        
        if uploaded_file is not None:
            # Tampilkan gambar asli
            image = Image.open(uploaded_file)
            
            with col1:
                st.subheader("🖼️ Gambar Asli")
                st.image(image, caption="Foto mobil yang diupload", use_column_width=True)
            
            if st.button("🔍 Analisis Kerusakan", type="primary", use_container_width=True):
                with st.spinner("Memproses segmentasi kerusakan..."):
                    annotated_img, damage_counts, severity = process_single_image(model, image, car_id)
                
                with col2:
                    st.subheader("🎯 Hasil Segmentasi")
                    st.image(annotated_img, caption="Mask overlay kerusakan", use_column_width=True)
                
                # Ringkasan hasil
                st.subheader("📊 Ringkasan Analisis")
                
                # Metrics dalam grid
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("🏷️ ID Mobil", car_id)
                with metric_cols[1]:
                    severity_color = {"Baik": "🟢", "Ringan": "🟡", "Sedang": "🟠", "Berat": "🔴"}
                    st.metric("⚡ Status", f"{severity_color.get(severity, '⚪')} {severity}")
                with metric_cols[2]:
                    st.metric("🔢 Total Kerusakan", sum(damage_counts.values()))
                with metric_cols[3]:
                    damaged_areas = sum(1 for count in damage_counts.values() if count > 0)
                    st.metric("📍 Area Terdampak", damaged_areas)
                
                # Detail per kategori
                st.subheader("🔍 Detail Kerusakan per Kategori")
                detail_cols = st.columns(3)
                col_idx = 0
                
                for damage, count in damage_counts.items():
                    if count > 0:
                        with detail_cols[col_idx % 3]:
                            color_rgb = DAMAGE_COLORS[damage]
                            color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                            st.markdown(f"""
                            <div style='padding: 10px; border-left: 4px solid {color_hex}; background-color: rgba{(*color_rgb, 0.1)}; margin: 5px 0;'>
                                <strong style='color: {color_hex};'>{damage}</strong><br>
                                <span style='font-size: 1.5em; font-weight: bold;'>{count}</span> area terdeteksi
                            </div>
                            """, unsafe_allow_html=True)
                        col_idx += 1
                
                # Tabel ringkasan
                df_single = pd.DataFrame([{
                    "ID_Mobil": car_id,
                    **damage_counts,
                    "Status": severity
                }])
                
                st.subheader("📋 Laporan Detail")
                st.dataframe(df_single, use_container_width=True)
                
                # Download hasil
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    # Download gambar hasil
                    img_pil = Image.fromarray(annotated_img)
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format='PNG')
                    st.download_button(
                        "💾 Download Gambar Hasil",
                        data=img_buffer.getvalue(),
                        file_name=f"{car_id}_segmented.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Download CSV
                    csv_buffer = io.StringIO()
                    df_single.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📄 Download Laporan CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{car_id}_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # Mode Batch Processing - Simplified
    elif mode == "📁 Batch Processing - Multiple Files":
        st.header("📁 Batch Processing - Analisis Multiple Files")
        st.markdown("Upload beberapa foto mobil sekaligus untuk dianalisis secara batch.")
        
        uploaded_files = st.file_uploader(
            "Pilih foto-foto mobil:",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Anda dapat memilih multiple files sekaligus. Format yang didukung: PNG, JPG, JPEG"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file siap diproses")
            
            # Preview beberapa file
            if len(uploaded_files) <= 5:
                st.subheader("👀 Preview File yang Dipilih:")
                preview_cols = st.columns(min(len(uploaded_files), 5))
                for i, file in enumerate(uploaded_files[:5]):
                    with preview_cols[i]:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_column_width=True)
            else:
                st.write(f"File pertama: {uploaded_files[0].name}")
                st.write(f"File terakhir: {uploaded_files[-1].name}")
                st.write("... dan lainnya")
            
            if st.button("🚀 Proses Semua File", type="primary", use_container_width=True):
                progress_container = st.container()
                
                with st.spinner("Memulai batch processing..."):
                    results_data = process_batch_images(model, uploaded_files, progress_container)
                
                if results_data:
                    st.success(f"✅ Berhasil memproses {len(results_data)} mobil!")
                    
                    # Statistik keseluruhan
                    st.subheader("📈 Ringkasan Batch")
                    severity_counts = {}
                    total_damages = 0
                    damage_breakdown = {damage: 0 for damage in DAMAGE_CLASSES.values()}
                    
                    for data in results_data.values():
                        severity = data["severity"]
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                        total_damages += sum(data["damage_counts"].values())
                        
                        for damage, count in data["damage_counts"].items():
                            damage_breakdown[damage] += count
                    
                    # Metrics overview
                    overview_cols = st.columns(4)
                    with overview_cols[0]:
                        st.metric("🚗 Total Mobil", len(results_data))
                    with overview_cols[1]:
                        st.metric("💥 Total Kerusakan", total_damages)
                    with overview_cols[2]:
                        good_cars = severity_counts.get("Baik", 0) + severity_counts.get("Ringan", 0)
                        st.metric("✅ Kondisi Baik/Ringan", good_cars)
                    with overview_cols[3]:
                        bad_cars = severity_counts.get("Sedang", 0) + severity_counts.get("Berat", 0)
                        st.metric("⚠️ Kondisi Sedang/Berat", bad_cars)
                    
                    # Breakdown kerusakan per kategori
                    st.subheader("📊 Breakdown Kerusakan per Kategori")
                    breakdown_cols = st.columns(3)
                    col_idx = 0
                    
                    for damage, total_count in damage_breakdown.items():
                        if total_count > 0:
                            with breakdown_cols[col_idx % 3]:
                                color_rgb = DAMAGE_COLORS[damage]
                                color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                                st.markdown(f"""
                                <div style='padding: 15px; border: 2px solid {color_hex}; border-radius: 10px; text-align: center; margin: 5px;'>
                                    <h3 style='color: {color_hex}; margin: 0;'>{damage}</h3>
                                    <h1 style='margin: 10px 0; color: {color_hex};'>{total_count}</h1>
                                    <small>total instances</small>
                                </div>
                                """, unsafe_allow_html=True)
                            col_idx += 1
                    
                    # Tabel laporan lengkap
                    st.subheader("📋 Laporan Lengkap")
                    df_report = create_damage_report(results_data)
                    st.dataframe(df_report, use_container_width=True)
                    
                    # Download section
                    st.subheader("💾 Download Hasil Batch")
                    
                    download_cols = st.columns(2)
                    
                    with download_cols[0]:
                        # Download CSV Report
                        csv_buffer = io.StringIO()
                        df_report.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "📄 Download Laporan CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"batch_damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with download_cols[1]:
                        # Download ZIP dengan semua hasil
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Tambahkan CSV report
                            zip_file.writestr("batch_report.csv", csv_buffer.getvalue())
                            
                            # Tambahkan semua gambar hasil
                            for car_id, data in results_data.items():
                                img_pil = Image.fromarray(data["annotated_image"])
                                img_buffer = io.BytesIO()
                                img_pil.save(img_buffer, format='PNG')
                                zip_file.writestr(f"segmented_images/{car_id}_segmented.png", img_buffer.getvalue())
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            "📦 Download Semua Hasil (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    
                    # Galeri hasil (sampel)
                    st.subheader("🖼️ Galeri Hasil Segmentasi")
                    st.markdown("Menampilkan beberapa contoh hasil segmentasi:")
                    
                    # Tampilkan maksimal 6 hasil dalam grid
                    sample_ids = list(results_data.keys())[:6]
                    
                    if len(sample_ids) <= 3:
                        gallery_cols = st.columns(len(sample_ids))
                    else:
                        gallery_cols = st.columns(3)
                    
                    for i, car_id in enumerate(sample_ids):
                        col_idx = i % 3
                        with gallery_cols[col_idx]:
                            data = results_data[car_id]
                            st.markdown(f"**{car_id}**")
                            st.markdown(f"Status: {data['severity']}")
                            st.image(data["annotated_image"], use_column_width=True)
                            
                            # Ringkasan kerusakan untuk sample ini
                            damage_summary = []
                            for damage, count in data["damage_counts"].items():
                                if count > 0:
                                    damage_summary.append(f"{damage}: {count}")
                            
                            if damage_summary:
                                st.caption(" | ".join(damage_summary))
                            else:
                                st.caption("Tidak ada kerusakan terdeteksi")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 50px;'>
        <p><strong>Sistem Identifikasi Kerusakan Mobil</strong></p>
        <p>Menggunakan YOLOv11 Instance Segmentation • Car Damage Dataset (CarDD)</p>
        <p><small>Dikembangkan dengan framework CRISP-DM untuk analisis kerusakan bodi kendaraan</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()