# app.py
import io
import os
import cv2
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import streamlit as st
from ultralytics import YOLO

# ==========================
# APP CONFIG
# ==========================
st.set_page_config(
    page_title="🚗 Car Damage Assessment System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# INFERENCE CONSTANTS (ubah di sini saja)
# ==========================
CONF_THRES = 0.15      # confidence threshold
IOU_THRES  = 0.45      # NMS IoU threshold
IMG_SIZE   = 640       # inference image size
MAX_DET    = 300       # max detections per image
DEVICE     = "auto"    # "auto" or "cpu"

# ==========================
# SEVERITY SYSTEM CONFIGURATION
# ==========================
# Bobot tingkat keparahan untuk setiap jenis kerusakan (0-10 scale)
DAMAGE_SEVERITY_WEIGHTS = {
    'dent': 3,           # Penyok - sedang
    'scratch': 2,        # Goresan - ringan  
    'crack': 6,          # Retak - berat
    'glass_damage': 8,   # Kerusakan kaca - sangat berat
    'lamp_damage': 7,    # Kerusakan lampu - berat
    'tire_damage': 9     # Kerusakan ban - sangat berat (safety critical)
}

# Threshold untuk kategori severity berdasarkan skor tertimbang
SEVERITY_THRESHOLDS = {
    'No Damage': (0, 0),      # Tidak ada kerusakan
    'Light': (0.1, 15),       # Ringan: skor 0.1-15
    'Medium': (15.1, 35),     # Sedang: skor 15.1-35  
    'Heavy': (35.1, 60),      # Berat: skor 35.1-60
    'Critical': (60.1, 999)   # Kritis: skor >60
}

# ==========================
# UI STYLES
# ==========================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 1.5rem; border-radius: 15px;
    text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.step-card { background: white; border: 2px solid #e9ecef; border-radius: 12px;
    padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.active-step { border-color: #667eea; background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%); }
.queue-item { background: #e8f5e8; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #28a745; }
.processing-item { background: #fff3cd; border-left: 4px solid #ffc107; padding: .75rem 1rem; border-radius: 10px; }

/* Updated severity styles */
.severity-light    { background: linear-gradient(135deg,#d4edda 0%,#c3e6cb 100%); color:#155724; border-left:4px solid #28a745; }
.severity-medium   { background: linear-gradient(135deg,#fff3cd 0%,#ffeaa7 100%); color:#856404; border-left:4px solid #ffc107; }
.severity-heavy    { background: linear-gradient(135deg,#f8d7da 0%,#f5c6cb 100%); color:#721c24; border-left:4px solid #dc3545; }
.severity-critical { background: linear-gradient(135deg,#d1ecf1 0%,#bee5eb 100%); color:#0c5460; border-left:4px solid #17a2b8; animation: pulse 2s infinite; }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(23, 162, 184, 0); }
    100% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0); }
}

.severity-score {
    background: #f8f9fa; border-radius: 8px; padding: 0.5rem 1rem; 
    font-weight: bold; display: inline-block; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# SESSION STATE INIT
# ==========================
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

# ==========================
# CLASS NAME MAP & COLORS
# ==========================
# Map nama kelas (sesuai model.names) → kunci internal
CLASS_KEY_MAP = {
    "Dent": "dent",
    "Scratch": "scratch",
    "Crack": "crack",
    "Glass Shatter": "glass_damage",
    "Lamp Broken": "lamp_damage",
    "Tire Flat": "tire_damage",
}
# Warna RGB untuk overlay
COLOR_MAP = {
    'dent': (255,107,107),
    'scratch': (78,205,196),
    'crack': (69,183,209),
    'glass_damage': (150,206,180),
    'lamp_damage': (255,234,167),
    'tire_damage': (221,160,221)
}

# ==========================
# MODEL LOADER + VALIDATION
# ==========================
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        # Validasi tipe task = segment
        task = getattr(model, 'task', None)
        if task is None:
            task = getattr(getattr(model, 'model', None), 'task', None)
        if task != 'segment':
            st.error("❌ Model bukan tipe *segmentation*. Latih/ekspor YOLOv11-SEG (task=segment) dan simpan sebagai `best.pt`.")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("📁 Pastikan file `best.pt` (YOLOv11-seg) ada di root.")
        return None

# ==========================
# ENHANCED SEVERITY CALCULATION
# ==========================
def calculate_weighted_severity_score(damage_counts: dict):
    """
    Menghitung skor severity berdasarkan bobot tingkat keparahan
    
    Returns:
        float: Weighted severity score
    """
    total_score = 0.0
    
    for damage_type, count in damage_counts.items():
        if count > 0:
            weight = DAMAGE_SEVERITY_WEIGHTS.get(damage_type, 1)
            # Skor = jumlah × bobot × faktor progresif
            # Faktor progresif: semakin banyak kerusakan sejenis, semakin parah
            progressive_factor = 1 + (count - 1) * 0.3  # +30% per kerusakan tambahan
            score = count * weight * progressive_factor
            total_score += score
    
    return round(total_score, 1)

def get_damage_severity(damage_counts: dict):
    """
    Menentukan kategori severity berdasarkan weighted score
    
    Returns:
        tuple: (severity_category, streamlit_status_type)
    """
    total_damages = sum(damage_counts.values())
    
    if total_damages == 0:
        return ("No Damage", "success", 0.0)
    
    weighted_score = calculate_weighted_severity_score(damage_counts)
    
    for category, (min_score, max_score) in SEVERITY_THRESHOLDS.items():
        if min_score <= weighted_score <= max_score:
            if category == "No Damage":
                return (category, "success", weighted_score)
            elif category == "Light":
                return (category, "success", weighted_score)
            elif category == "Medium":
                return (category, "warning", weighted_score)
            elif category == "Heavy":
                return (category, "error", weighted_score)
            elif category == "Critical":
                return (category, "info", weighted_score)  # Using info for critical (blue)
    
    # Fallback untuk skor sangat tinggi
    return ("Critical", "info", weighted_score)

def get_damage_breakdown(damage_counts: dict):
    """
    Memberikan breakdown kontribusi setiap jenis kerusakan terhadap severity
    
    Returns:
        list: List of tuples (damage_type, count, individual_score, percentage)
    """
    breakdown = []
    total_score = calculate_weighted_severity_score(damage_counts)
    
    if total_score == 0:
        return breakdown
    
    for damage_type, count in damage_counts.items():
        if count > 0:
            weight = DAMAGE_SEVERITY_WEIGHTS.get(damage_type, 1)
            progressive_factor = 1 + (count - 1) * 0.3
            individual_score = count * weight * progressive_factor
            percentage = (individual_score / total_score) * 100
            
            breakdown.append({
                'damage_type': damage_type.replace('_', ' ').title(),
                'count': count,
                'weight': weight,
                'individual_score': round(individual_score, 1),
                'percentage': round(percentage, 1)
            })
    
    # Sort by individual score (descending)
    breakdown.sort(key=lambda x: x['individual_score'], reverse=True)
    return breakdown

def process_image_segmentation(model, image: Image.Image):
    """Run segmentation + overlay mask; return (annotated_pil, counts_dict)"""
    img_np = np.array(image)

    results = model(
        img_np,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        max_det=MAX_DET,
        device=None if DEVICE == "auto" else DEVICE,
        verbose=False
    )

    counts = {'dent':0,'scratch':0,'crack':0,'glass_damage':0,'lamp_damage':0,'tire_damage':0}
    draw = img_np.copy()
    H, W = draw.shape[:2]
    alpha = 0.45

    if results and len(results) > 0:
        r = results[0]
        if getattr(r, "masks", None) is None or r.masks is None:
            return Image.fromarray(draw), counts

        masks = r.masks.data.cpu().numpy()          # [N,h,w]
        xyxy  = r.boxes.xyxy.cpu().numpy()
        cls   = r.boxes.cls.cpu().numpy().astype(int)
        conf  = r.boxes.conf.cpu().numpy()
        names = model.names  # dict

        for i in range(len(cls)):
            if conf[i] < CONF_THRES:
                continue
            label_name = names.get(int(cls[i]), "unknown")
            key = CLASS_KEY_MAP.get(label_name)
            if key is None:
                # fallback: coba cocokkan lowercase sederhana
                ln = label_name.lower()
                if "dent" in ln: key = "dent"
                elif "scratch" in ln: key = "scratch"
                elif "crack" in ln: key = "crack"
                elif "glass" in ln: key = "glass_damage"
                elif "lamp" in ln or "head" in ln: key = "lamp_damage"
                elif "tire" in ln or "wheel" in ln: key = "tire_damage"
                else:
                    continue

            counts[key] += 1

            # Resize mask ke ukuran gambar & binarisasi
            m = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
            m_bin = m > 0.5

            # Overlay warna
            color = COLOR_MAP[key]
            overlay = np.zeros_like(draw)
            overlay[m_bin] = color
            draw = cv2.addWeighted(draw, 1 - alpha, overlay, alpha, 0)

            # Box + label
            x1, y1, x2, y2 = map(int, xyxy[i])
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                draw, f"{label_name}",
                (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    return Image.fromarray(draw), counts

def create_summary_report(cars: list):
    rows = []
    for car in cars:
        rows.append({
            'License_Plate': car['plate'],
            'Total_Images': len(car['images']),
            'Dent': car['total_damages']['dent'],
            'Scratch': car['total_damages']['scratch'],
            'Crack': car['total_damages']['crack'],
            'Glass_Damage': car['total_damages']['glass_damage'],
            'Lamp_Damage': car['total_damages']['lamp_damage'],
            'Tire_Damage': car['total_damages']['tire_damage'],
            'Total_Damages': sum(car['total_damages'].values()),
            'Severity_Score': car['severity'][2],  # Weighted score
            'Severity': car['severity'][0],
            'Processing_Date': car['processed_time'].strftime('%Y-%m-%d %H:%M:%S'),
        })
    return pd.DataFrame(rows)

def create_results_zip(processed_cars: list) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Summary
        summary_df = create_summary_report(processed_cars)
        zf.writestr("summary_report.csv", summary_df.to_csv(index=False))

        # Detailed per image
        detailed = []
        for car in processed_cars:
            for img in car['images']:
                img_severity = get_damage_severity(img['damages'])
                detailed.append({
                    'License_Plate': car['plate'],
                    'Image_Name': img['name'],
                    'Dent': img['damages']['dent'],
                    'Scratch': img['damages']['scratch'],
                    'Crack': img['damages']['crack'],
                    'Glass_Damage': img['damages']['glass_damage'],
                    'Lamp_Damage': img['damages']['lamp_damage'],
                    'Tire_Damage': img['damages']['tire_damage'],
                    'Total_in_Image': sum(img['damages'].values()),
                    'Image_Severity_Score': img_severity[2],
                    'Image_Severity': img_severity[0],
                    'Car_Severity_Score': car['severity'][2],
                    'Car_Severity': car['severity'][0],
                    'Processing_Time': car['processed_time'].strftime('%Y-%m-%d %H:%M:%S')
                })
        detailed_df = pd.DataFrame(detailed)
        zf.writestr("detailed_per_image_report.csv", detailed_df.to_csv(index=False))

        # Images
        for car in processed_cars:
            car_folder = f"{car['plate'].replace(' ', '_')}/"
            for img in car['images']:
                buf1 = io.BytesIO(); img['original'].save(buf1, format='PNG')
                zf.writestr(f"{car_folder}original_{img['name']}", buf1.getvalue())
                buf2 = io.BytesIO(); img['annotated'].save(buf2, format='PNG')
                zf.writestr(f"{car_folder}segmented_{img['name']}", buf2.getvalue())

    buffer.seek(0)
    return buffer.getvalue()

# ==========================
# UI FLOW
# ==========================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Car Damage Assessment System</h1>
        <p>AI-powered vehicle damage detection using YOLOv11 Instance Segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if not model:
        st.stop()

    # Show severity configuration in sidebar
    with st.sidebar:
        st.subheader("⚖️ Severity Configuration")
        st.markdown("**Damage Severity Weights:**")
        for damage, weight in DAMAGE_SEVERITY_WEIGHTS.items():
            st.write(f"• {damage.replace('_', ' ').title()}: {weight}/10")
        
        st.markdown("---")
        st.markdown("**Severity Thresholds:**")
        for category, (min_score, max_score) in SEVERITY_THRESHOLDS.items():
            if category == "No Damage":
                continue
            st.write(f"• {category}: {min_score}-{max_score if max_score != 999 else '∞'}")

    step = st.session_state.current_step
    if step == 'dashboard': show_dashboard()
    elif step == 'input': show_input_step()
    elif step == 'queue': show_queue_step()
    elif step == 'process': show_process_step(model)
    elif step == 'results': show_results_step()
    elif step == 'download': show_download_step()

def show_dashboard():
    st.markdown('<div class="step-card active-step"><h2>📊 Halaman Dashboard</h2><p>Selamat datang di sistem identifikasi kerusakan mobil menggunakan AI dengan Weighted Severity System</p></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🚗 Cars in Queue", len(st.session_state.car_queue))
    total_processed = len(st.session_state.processed_results)
    c2.metric("✅ Cars Processed", total_processed)
    
    if total_processed > 0:
        total_damages = sum(sum(c['total_damages'].values()) for c in st.session_state.processed_results)
        avg_severity_score = sum(c['severity'][2] for c in st.session_state.processed_results) / total_processed
        c3.metric("🔍 Total Damages Found", total_damages)
        c4.metric("⚖️ Avg Severity Score", f"{avg_severity_score:.1f}")
    else:
        c3.metric("🔍 Total Damages Found", 0)
        c4.metric("⚖️ Avg Severity Score", "0.0")

    st.markdown("---")
    
    if st.session_state.processed_results:
        st.subheader("📈 Recent Processing Results")
        
        # Group by severity for better overview
        severity_groups = {}
        for car in st.session_state.processed_results:
            severity = car['severity'][0]
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(car)
        
        # Show severity distribution
        st.subheader("📊 Severity Distribution")
        col1, col2, col3, col4 = st.columns(4)
        
        light_count = len(severity_groups.get('Light', []))
        medium_count = len(severity_groups.get('Medium', []))
        heavy_count = len(severity_groups.get('Heavy', []))
        critical_count = len(severity_groups.get('Critical', []))
        
        col1.metric("🟢 Light", light_count)
        col2.metric("🟡 Medium", medium_count) 
        col3.metric("🔴 Heavy", heavy_count)
        col4.metric("🔵 Critical", critical_count)
        
        st.markdown("---")
        
        # Show recent results with enhanced info
        for car in st.session_state.processed_results[-3:]:
            sev = car['severity'][0].lower()
            score = car['severity'][2]
            total_dmg = sum(car['total_damages'].values())
            
            # Get most severe damage type
            breakdown = get_damage_breakdown(car['total_damages'])
            top_damage = breakdown[0]['damage_type'] if breakdown else "None"
            
            st.markdown(f'''
            <div class="processing-item severity-{sev}">
                <strong>🚗 {car["plate"]}</strong><br>
                <div class="severity-score">Severity Score: {score}</div><br>
                📊 {car["severity"][0]} | 🔍 {total_dmg} damages | 🎯 Primary: {top_damage}
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("---")
    _, c, _ = st.columns([1,2,1])
    with c:
        if st.button("🚀 Mulai Identifikasi Kerusakan", use_container_width=True, type="primary"):
            st.session_state.current_step = 'input'
            st.rerun()

def show_input_step():
    st.markdown('<div class="step-card active-step"><h2>📝 Input Nomor Plat & Gambar</h2><p>Masukkan nomor plat dan upload gambar mobil untuk dianalisis</p></div>', unsafe_allow_html=True)

    c1, _ = st.columns([1,4])
    with c1:
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.current_step = 'dashboard'; st.rerun()

    st.subheader("🚗 Detail Kendaraan")
    plate = st.text_input("Nomor Plat Kendaraan", value=st.session_state.current_license_plate, placeholder="Contoh: B 1234 CD")
    st.session_state.current_license_plate = plate

    st.subheader("📸 Upload Gambar")
    files = st.file_uploader("Pilih gambar kendaraan (bisa multiple)", type=['png','jpg','jpeg'], accept_multiple_files=True)

    if files:
        st.session_state.uploaded_images = []
        cols = st.columns(min(len(files), 3))
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            st.session_state.uploaded_images.append({'name': f.name, 'image': img})
            with cols[i % 3]:
                st.image(img, caption=f.name, use_container_width=True)
        st.success(f"✅ {len(files)} gambar telah diupload")

    st.markdown("---")
    _, c2, _ = st.columns([2,2,1])
    with c2:
        can_go = plate.strip() and st.session_state.uploaded_images
        if st.button("➡️ Tambah ke Queue", disabled=not can_go, use_container_width=True):
            st.session_state.car_queue.append({'plate': plate.strip(), 'images': st.session_state.uploaded_images.copy(), 'timestamp': datetime.now()})
            st.session_state.current_license_plate = ""; st.session_state.uploaded_images = []
            st.session_state.current_step = 'queue'; st.rerun()

def show_queue_step():
    st.markdown('<div class="step-card active-step"><h2>📋 Tambah Queue</h2><p>Kelola antrian kendaraan yang akan diproses</p></div>', unsafe_allow_html=True)

    c1, _ = st.columns([1,4])
    with c1:
        if st.button("⬅️ Back to Input"):
            st.session_state.current_step = 'input'; st.rerun()

    if st.session_state.car_queue:
        st.subheader(f"🚗 Antrian Kendaraan ({len(st.session_state.car_queue)} mobil)")
        for i, car in enumerate(st.session_state.car_queue):
            st.markdown(f'<div class="queue-item"><strong>🚗 {car["plate"]}</strong><br>📸 {len(car["images"])} gambar | 🕐 {car["timestamp"].strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
            _, c2 = st.columns([4,1])
            with c2:
                if st.button("❌ Remove", key=f"rm_{i}"):
                    st.session_state.car_queue.pop(i); st.rerun()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("➕ Tambah Mobil Lain", use_container_width=True):
                st.session_state.current_step = 'input'; st.rerun()
        with c2:
            if st.button("🚀 Proses Semua", use_container_width=True, type="primary"):
                st.session_state.current_step = 'process'; st.rerun()
        with c3:
            if st.button("🗑️ Kosongkan Queue", use_container_width=True):
                st.session_state.car_queue = []; st.rerun()
    else:
        st.info("📭 Queue kosong. Tambahkan mobil terlebih dahulu.")
        _, c2, _ = st.columns([1,2,1])
        with c2:
            if st.button("➕ Tambah Mobil Pertama", use_container_width=True):
                st.session_state.current_step = 'input'; st.rerun()

def show_process_step(model):
    st.markdown('<div class="step-card active-step"><h2>⚙️ Proses Semua</h2><p>Sedang memproses semua kendaraan dalam antrian...</p></div>', unsafe_allow_html=True)

    if not st.session_state.car_queue:
        st.error("❌ Queue kosong!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'; st.rerun()
        return

    progress = st.progress(0); status = st.empty(); result_box = st.container()

    processed = []; total = len(st.session_state.car_queue)
    for i, car in enumerate(st.session_state.car_queue):
        status.text(f"🔍 Processing {car['plate']} ({i+1}/{total})...")

        totals = {'dent':0,'scratch':0,'crack':0,'glass_damage':0,'lamp_damage':0,'tire_damage':0}
        imgs = []

        for j, item in enumerate(car['images']):
            status.text(f"🔍 {car['plate']} — Image {j+1}/{len(car['images'])}")
            annotated, counts = process_image_segmentation(model, item['image'])
            for k,v in counts.items(): totals[k] += v
            imgs.append({'name': item['name'], 'original': item['image'], 'annotated': annotated, 'damages': counts})

        severity = get_damage_severity(totals)
        processed.append({
            'plate': car['plate'], 
            'images': imgs, 
            'total_damages': totals, 
            'severity': severity, 
            'processed_time': datetime.now()
        })

        progress.progress((i+1)/total)
        sev_cls = severity[0].lower()
        score = severity[2]
        result_box.markdown(f'''
        <div class="processing-item severity-{sev_cls}">
            <strong>✅ {car["plate"]}</strong><br>
            <div class="severity-score">Score: {score}</div><br>
            {severity[0]} damage | Total kerusakan: {sum(totals.values())}
        </div>
        ''', unsafe_allow_html=True)

    st.session_state.processed_results = processed
    st.session_state.car_queue = []
    status.text("✅ Semua kendaraan berhasil diproses!")
    st.success(f"🎉 Berhasil memproses {total} kendaraan!")

    import time; time.sleep(1.2)
    st.session_state.current_step = 'results'; st.rerun()

def show_results_step():
    st.markdown('<div class="step-card active-step"><h2>📊 Hasil Analisis dengan Weighted Severity</h2><p>Hasil analisis kerusakan dengan sistem severity berbobot</p></div>', unsafe_allow_html=True)

    if not st.session_state.processed_results:
        st.error("❌ Tidak ada hasil untuk ditampilkan!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'; st.rerun()
        return

    st.subheader("📈 Ringkasan Hasil")
    total_cars = len(st.session_state.processed_results)
    total_damages = sum(sum(c['total_damages'].values()) for c in st.session_state.processed_results)
    avg_score = sum(c['severity'][2] for c in st.session_state.processed_results) / total_cars if total_cars > 0 else 0
    critical_cars = sum(1 for c in st.session_state.processed_results if c['severity'][0] == 'Critical')
    
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🚗 Total Mobil", total_cars)
    c2.metric("🔍 Total Kerusakan", total_damages)
    c3.metric("⚖️ Rata-rata Score", f"{avg_score:.1f}")
    c4.metric("🆘 Critical Cases", critical_cars)

    st.markdown("---")

    for car in st.session_state.processed_results:
        severity_name, _, severity_score = car['severity']
        st.subheader(f"🚗 {car['plate']} — {severity_name} Damage")
        
        # Show severity breakdown
        breakdown = get_damage_breakdown(car['total_damages'])
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"""
            <div class="severity-score">
                Severity Score: <strong>{severity_score}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if breakdown:
                st.write("**🎯 Damage Breakdown:**")
                for item in breakdown:
                    st.write(f"• {item['damage_type']}: {item['count']} × (weight: {item['weight']}) = {item['individual_score']} pts ({item['percentage']:.1f}%)")
        
        with col2:
            damages = car['total_damages']
            found = [f"{k.replace('_',' ').title()}: {v}" for k,v in damages.items() if v>0]
            st.write("**🔍 Kerusakan ditemukan:** " + (", ".join(found) if found else "Tidak ada"))
            
            # Show risk assessment
            if severity_score >= 60:
                st.error("🆘 **CRITICAL**: Kendaraan memerlukan perbaikan segera!")
            elif severity_score >= 35:
                st.warning("⚠️ **HIGH RISK**: Perbaikan diperlukan dalam waktu dekat")
            elif severity_score >= 15:
                st.info("ℹ️ **MEDIUM RISK**: Monitor dan pertimbangkan perbaikan")
            else:
                st.success("✅ **LOW RISK**: Kerusakan minor atau kosmetik")

        st.markdown("---")

        for img in car['images']:
            st.write(f"**📸 {img['name']}**")
            
            # Calculate image-level severity
            img_severity = get_damage_severity(img['damages'])
            img_breakdown = get_damage_breakdown(img['damages'])
            
            # Show image severity info
            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.markdown(f"""
                <div class="severity-score">
                    Image Score: <strong>{img_severity[2]}</strong><br>
                    Level: <strong>{img_severity[0]}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                if img_breakdown:
                    st.write("**Image damage contribution:**")
                    for item in img_breakdown[:3]:  # Show top 3
                        st.write(f"• {item['damage_type']}: {item['individual_score']} pts")
            
            # Show images side by side
            a, b = st.columns(2)
            with a: 
                st.image(img['original'], caption="Gambar Asli", use_container_width=True)
            with b: 
                st.image(img['annotated'], caption="Hasil Deteksi dengan Mask", use_container_width=True)

            # Show damage details for this image
            if sum(img['damages'].values()) > 0:
                st.write("**📊 Kerusakan pada gambar ini:**")
                cols = st.columns(6)
                for j, (k, v) in enumerate(img['damages'].items()):
                    if v > 0: 
                        weight = DAMAGE_SEVERITY_WEIGHTS.get(k, 1)
                        cols[j%6].metric(k.replace('_',' ').title(), f"{v} (w:{weight})")

        st.markdown("---")

    # Overall statistics
    st.subheader("📊 Statistical Summary")
    
    # Create severity distribution chart data
    severity_dist = {}
    score_ranges = {
        "No Damage (0)": 0,
        "Light (0.1-15)": 0, 
        "Medium (15.1-35)": 0,
        "Heavy (35.1-60)": 0,
        "Critical (60+)": 0
    }
    
    for car in st.session_state.processed_results:
        score = car['severity'][2]
        if score == 0:
            score_ranges["No Damage (0)"] += 1
        elif score <= 15:
            score_ranges["Light (0.1-15)"] += 1
        elif score <= 35:
            score_ranges["Medium (15.1-35)"] += 1
        elif score <= 60:
            score_ranges["Heavy (35.1-60)"] += 1
        else:
            score_ranges["Critical (60+)"] += 1
    
    # Display distribution
    st.write("**Severity Distribution:**")
    cols = st.columns(5)
    for i, (range_name, count) in enumerate(score_ranges.items()):
        with cols[i]:
            percentage = (count / total_cars * 100) if total_cars > 0 else 0
            st.metric(range_name, f"{count} ({percentage:.1f}%)")

    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Prediksi Lagi", use_container_width=True):
            st.session_state.processed_results = []
            st.session_state.current_step = 'dashboard'; st.rerun()
    with c2:
        if st.button("📥 Download ZIP?", use_container_width=True, type="primary"):
            st.session_state.current_step = 'download'; st.rerun()
    with c3:
        if st.button("📊 Tampilkan Output", use_container_width=True):
            show_detailed_output()

def show_download_step():
    st.markdown('<div class="step-card active-step"><h2>📥 File Download</h2><p>Download hasil analisis dalam berbagai format</p></div>', unsafe_allow_html=True)
    if not st.session_state.processed_results:
        st.error("❌ Tidak ada hasil untuk didownload!")
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state.current_step = 'dashboard'; st.rerun()
        return

    st.subheader("📊 Enhanced Summary Report")
    summary_df = create_summary_report(st.session_state.processed_results)
    st.dataframe(summary_df, use_container_width=True)

    # Show severity statistics
    st.subheader("📈 Severity Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Statistics:**")
        scores = [car['severity'][2] for car in st.session_state.processed_results]
        if scores:
            st.write(f"• Average Score: {np.mean(scores):.1f}")
            st.write(f"• Median Score: {np.median(scores):.1f}")
            st.write(f"• Max Score: {max(scores):.1f}")
            st.write(f"• Min Score: {min(scores):.1f}")
    
    with col2:
        st.write("**Risk Distribution:**")
        severity_counts = {}
        for car in st.session_state.processed_results:
            sev = car['severity'][0]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        for sev, count in severity_counts.items():
            percentage = (count / len(st.session_state.processed_results)) * 100
            st.write(f"• {sev}: {count} ({percentage:.1f}%)")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download Enhanced CSV Report",
            data=summary_df.to_csv(index=False),
            file_name=f"enhanced_car_damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True
        )
    with c2:
        st.download_button(
            "📦 Download ZIP with Images & Analysis",
            data=create_results_zip(st.session_state.processed_results),
            file_name=f"enhanced_car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip", use_container_width=True
        )

    st.markdown("---")
    a, b = st.columns(2)
    with a:
        if st.button("⬅️ Kembali ke Hasil", use_container_width=True):
            st.session_state.current_step = 'results'; st.rerun()
    with b:
        if st.button("🏠 Kembali ke Dashboard", use_container_width=True):
            st.session_state.current_step = 'dashboard'; st.rerun()

def show_detailed_output():
    st.subheader("📋 Detailed Output Table with Severity Scores")
    rows = []
    for car in st.session_state.processed_results:
        for img in car['images']:
            img_severity = get_damage_severity(img['damages'])
            rows.append({
                'License_Plate': car['plate'],
                'Image_Name': img['name'],
                'Dent': img['damages']['dent'],
                'Scratch': img['damages']['scratch'],
                'Crack': img['damages']['crack'],
                'Glass_Damage': img['damages']['glass_damage'],
                'Lamp_Damage': img['damages']['lamp_damage'],
                'Tire_Damage': img['damages']['tire_damage'],
                'Total_in_Image': sum(img['damages'].values()),
                'Image_Score': img_severity[2],
                'Image_Severity': img_severity[0],
                'Car_Total_Score': car['severity'][2],
                'Car_Severity': car['severity'][0]
            })
    
    detailed_df = pd.DataFrame(rows)
    st.dataframe(detailed_df, use_container_width=True)
    
    # Add download button for detailed report
    st.download_button(
        "📥 Download Detailed Report",
        data=detailed_df.to_csv(index=False),
        file_name=f"detailed_severity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()