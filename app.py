import os
import io
import cv2
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="Car Damage Detection", layout="wide", page_icon="🚗")
st.title("🚗 Car Damage Detection - YOLOv11")

# ==========================
# Constants (Fixed params)
# ==========================
WEIGHTS_FILE = "best.pt"
FIXED_CONF = 0.15
FIXED_IOU = 0.7
FIXED_IMGSZ = 640

# Severity thresholds
SEVERITY_T1 = 0.25  # <25% = Light
SEVERITY_T2 = 0.60  # 25–60% = Medium, >60% = Heavy

# ==========================
# Real-time auto-start (HARD-CODED CAMERA INDEX 0)
# ==========================
AUTO_START_RT = True
DEFAULT_CAMERA_SRC = "0"        # hanya pakai webcam index 0
FALLBACK_CAMERA_SRCS = ["0"]    # tidak ada fallback lain

# (opsional) set opsi FFmpeg agar stabil bila kelak pakai RTSP
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|buffer_size;4096"
)

# ==========================
# Utility Functions
# ==========================
def to_pil_rgb(arr_bgr_or_rgb):
    if arr_bgr_or_rgb is None:
        return None
    a = arr_bgr_or_rgb
    if a.ndim == 3 and a.shape[2] == 3:
        a = a[:, :, ::-1].copy()
    return Image.fromarray(a)

def compute_severity(mask_bin: np.ndarray, xyxy: np.ndarray):
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
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def plot_custom_overlay(pil_img: Image.Image, boxes, masks, names_map):
    if boxes is None or len(boxes) == 0:
        return pil_img
    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    # masks
    if masks is not None and masks.data is not None:
        mask_data = masks.data.cpu().numpy()
        for i, mask in enumerate(mask_data):
            if i < len(cls):
                color = colors[int(cls[i]) % len(colors)]
                mask_resized = cv2.resize(mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]))
                mask_colored = np.zeros_like(img_bgr); mask_colored[:, :] = color
                alpha = 0.3
                mask_bool = mask_resized > 0.5
                img_bgr[mask_bool] = (1 - alpha) * img_bgr[mask_bool] + alpha * mask_colored[mask_bool]

    # boxes + label (tanpa conf)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        cls_id = int(cls[i]); cls_name = names_map.get(cls_id, str(cls_id))
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = cls_name
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_left = (x1, max(0, y1 - th - bl - 5)); bottom_right = (x1 + tw, y1)
        cv2.rectangle(img_bgr, top_left, bottom_right, color, -1)
        cv2.putText(img_bgr, label, (x1, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def create_summary_text(plate, class_name, severity):
    if class_name == "no_detection":
        return f"Mobil dengan nomor plat {plate} tidak terdeteksi mengalami kerusakan."
    return f"Mobil dengan nomor plat {plate} terdeteksi mengalami kerusakan {class_name} dengan tingkat keparahan {severity}."

# ==========================
# Real-Time Helpers
# ==========================
def open_video_capture(src):
    try:
        use_ffmpeg = isinstance(src, str) and (src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://"))
        if use_ffmpeg:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cam_index = None
            if isinstance(src, str):
                try: cam_index = int(src)
                except Exception: cam_index = None
            cap = cv2.VideoCapture(cam_index if cam_index is not None else src)
        if not cap.isOpened():
            return None, f"Tidak bisa membuka sumber video: {src}"
        return cap, None
    except Exception as e:
        return None, f"Gagal membuka video ({src}): {e}"

def ensure_session_flags():
    if "rt_streaming" not in st.session_state:
        st.session_state.rt_streaming = False
    if "rt_last_frame_time" not in st.session_state:
        st.session_state.rt_last_frame_time = None
    if "rt_src" not in st.session_state:
        st.session_state.rt_src = DEFAULT_CAMERA_SRC
    if "rt_auto_started" not in st.session_state:
        st.session_state.rt_auto_started = False

# ==========================
# Model Loading
# ==========================
@st.cache_resource(show_spinner=True)
def load_model_from_path():
    try:
        if Path(WEIGHTS_FILE).exists():
            return YOLO(WEIGHTS_FILE)
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model_from_path()
if model is None:
    st.error("❌ **Model file tidak ditemukan!**")
    st.info(f"Letakkan `{WEIGHTS_FILE}` di direktori aplikasi lalu restart.")
    st.stop()
st.success(f"✅ **Model loaded:** `{WEIGHTS_FILE}`")

# ==========================
# Inference (Single Image)
# ==========================
def run_inference_on_image(model, pil_img: Image.Image, conf=FIXED_CONF, iou=FIXED_IOU, imgsz=FIXED_IMGSZ):
    results = model.predict(source=pil_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]
    names_map = r.names
    boxes = getattr(r, "boxes", None)
    masks = getattr(r, "masks", None)

    overlay_pil = plot_custom_overlay(pil_img, boxes, masks, names_map)

    records = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        mask_np = None
        if masks is not None and masks.data is not None:
            m = masks.data  # (N,H,W)
            mask_np = (m.cpu().numpy() > 0.5).astype(np.uint8)

        for i in range(len(xyxy)):
            cls_id = int(cls[i]); cls_name = names_map.get(cls_id, str(cls_id))
            conf_i = float(confs[i]); xyxy_i = xyxy[i]
            if mask_np is not None and i < mask_np.shape[0]:
                mask_area, bbox_area, ratio, severity = compute_severity(mask_np[i], xyxy_i)
            else:
                bbox_area = max(1, int((xyxy_i[2]-xyxy_i[0])*(xyxy_i[3]-xyxy_i[1])))
                mask_area, ratio, severity = 0, 0.0, "Light"
            records.append({
                "class_id": cls_id, "class_name": cls_name, "confidence": conf_i,
                "x1": int(xyxy_i[0]), "y1": int(xyxy_i[1]), "x2": int(xyxy_i[2]), "y2": int(xyxy_i[3]),
                "mask_area": int(mask_area), "bbox_area": int(bbox_area),
                "area_ratio": float(ratio), "severity": severity
            })
    return overlay_pil, records

# ==========================
# Session State (Batch Queue)
# ==========================
if "entries" not in st.session_state:
    st.session_state.entries = []
if "input_plate" not in st.session_state:
    st.session_state.input_plate = ""
if "clear_inputs" not in st.session_state:
    st.session_state.clear_inputs = False

# ==========================
# Sidebar - Input Only
# ==========================
with st.sidebar:
    st.header("📝 Add Vehicle")
    if st.session_state.clear_inputs:
        st.session_state.input_plate = ""
        st.session_state.clear_inputs = False
        st.rerun()

    plate = st.text_input(
        "Plate Number",
        value=st.session_state.input_plate,
        placeholder="B 1234 ABC",
        max_chars=11,
        help="Format: [A-Z] [1–4 digits] [A-Z][A-Z][A-Z]"
    )

    files = st.file_uploader(
        "Upload Images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key=f"file_uploader_{len(st.session_state.entries)}"
    )

    if st.button("➕ Add to Queue", use_container_width=True):
        if not plate:
            st.warning("Masukkan nomor plat terlebih dahulu")
        elif not files:
            st.warning("Upload minimal 1 gambar")
        else:
            packed_files = [(f.name, f.read()) for f in files]
            st.session_state.entries.append({"plate": plate.upper().strip(), "files": packed_files})
            st.success(f"Ditambahkan: {plate.upper()} ({len(packed_files)} gambar)")
            st.session_state.clear_inputs = True
            st.rerun()

    st.divider()
    if st.session_state.entries:
        st.subheader("📋 Processing Queue")
        for entry in st.session_state.entries:
            st.text(f"• {entry['plate']} — {len(entry['files'])} gambar")
        if st.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state.entries = []
            st.session_state.clear_inputs = True
            st.success("Queue dikosongkan!")
            st.rerun()

# ==========================
# Tabs
# ==========================
tab_batch, tab_realtime = st.tabs(["📦 Batch Processing", "🟢 Real-Time Scan"])

# --------------------------
# Batch Processing
# --------------------------
with tab_batch:
    if not st.session_state.entries:
        st.info("👆 Tambahkan kendaraan di sidebar, lalu kembali ke tab ini untuk **Process All**.")
        st.header("📖 Informasi")
        st.markdown("""
        **Severity Levels:**
        - 🟢 **Light**: < 25% area
        - 🟡 **Medium**: 25–60% area
        - 🔴 **Heavy**: > 60% area
        """)
    else:
        st.header(f"🚀 Ready to Process {len(st.session_state.entries)} Vehicle(s)")
        total_images = sum(len(e['files']) for e in st.session_state.entries)
        st.metric("Total Images to Process", total_images)

        if st.button("🚀 Process All", type="primary", use_container_width=True):
            all_records, summary_records = [], []
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                orig_dir = tmp_root / "original"
                seg_dir = tmp_root / "segmented"
                orig_dir.mkdir(parents=True, exist_ok=True)
                seg_dir.mkdir(parents=True, exist_ok=True)

                processed = 0
                for entry in st.session_state.entries:
                    plate = entry["plate"]
                    st.subheader(f"🚗 Processing: {plate}")

                    for file_idx, (filename, file_bytes) in enumerate(entry["files"], 1):
                        processed += 1
                        status_text.text(f"Processing {plate} — {filename} ({processed}/{total_images})")

                        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                        overlay_pil, detections = run_inference_on_image(model, pil_img)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(pil_img, caption=f"Original - {filename}", use_container_width=True)
                        with c2:
                            st.image(overlay_pil or pil_img, caption=f"Detection - {filename}", use_container_width=True)

                        safe_plate = plate.replace(" ", "_")
                        orig_name = f"{safe_plate}_{file_idx:02d}_{Path(filename).name}"
                        seg_name = f"{safe_plate}_{file_idx:02d}_{Path(filename).stem}_detected.jpg"
                        (orig_dir / orig_name).write_bytes(bytes_from_pil(pil_img, "JPEG"))
                        (seg_dir / seg_name).write_bytes(bytes_from_pil(overlay_pil or pil_img, "JPEG"))

                        if detections:
                            for det_idx, det in enumerate(detections, 1):
                                all_records.append({"plate": plate, "image": orig_name, "detection_id": det_idx, **det})
                                summary_records.append({
                                    "plate": plate, "image": filename,
                                    "summary": create_summary_text(plate, det["class_name"], det["severity"])
                                })
                        else:
                            all_records.append({
                                "plate": plate, "image": orig_name, "detection_id": 0,
                                "class_id": -1, "class_name": "no_detection", "confidence": 0.0,
                                "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                                "mask_area": 0, "bbox_area": 0, "area_ratio": 0.0, "severity": "None"
                            })
                            summary_records.append({
                                "plate": plate, "image": filename,
                                "summary": create_summary_text(plate, "no_detection", "None")
                            })

                        progress_bar.progress(processed / total_images)
                    st.divider()

                df_detailed = pd.DataFrame(all_records)
                df_summary = pd.DataFrame(summary_records)

                csv_detailed_path = tmp_root / "detection_results_detailed.csv"
                csv_summary_path = tmp_root / "detection_results_summary.csv"
                df_detailed.to_csv(csv_detailed_path, index=False)
                df_summary.to_csv(csv_summary_path, index=False)

                st.header("📋 Final Results Summary")
                st.success(f"✅ Processing complete! Found {len(df_detailed)} total detections.")
                if len(df_detailed) > 0 and (df_detailed['class_id'] != -1).any():
                    damage_summary = df_detailed[df_detailed['class_id'] != -1].groupby('class_name').agg({
                        'detection_id': 'count',
                        'severity': lambda x: x.value_counts().to_dict()
                    }).rename(columns={'detection_id': 'count'})
                    st.dataframe(damage_summary, use_container_width=True)

                st.subheader("📄 Summary Report")
                st.dataframe(df_summary, use_container_width=True)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(str(csv_detailed_path), arcname="detection_results_detailed.csv")
                    zf.write(str(csv_summary_path), arcname="detection_results_summary.csv")
                    for p in orig_dir.rglob("*"):
                        if p.is_file(): zf.write(str(p), arcname=f"original/{p.name}")
                    for p in seg_dir.rglob("*"):
                        if p.is_file(): zf.write(str(p), arcname=f"segmented/{p.name}")
                zip_buffer.seek(0)

                st.download_button(
                    label="⬇️ Download Results (ZIP)",
                    data=zip_buffer,
                    file_name=f"car_damage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            status_text.empty()
            progress_bar.empty()

# --------------------------
# Real-Time Scan (Auto-Start, camera index 0)
# --------------------------
with tab_realtime:
    st.header("🟢 Real-Time Scan")
    ensure_session_flags()

    # Auto-start saat tab dibuka
    if AUTO_START_RT and not st.session_state.rt_auto_started and not st.session_state.rt_streaming:
        chosen = None; last_err = None
        for candidate in FALLBACK_CAMERA_SRCS:
            cap_test, err = open_video_capture(candidate)
            if err is None:
                st.session_state.rt_src = candidate
                cap_test.release()
                chosen = candidate
                break
            last_err = err
        if chosen is not None:
            st.session_state.rt_streaming = True
            st.session_state.rt_auto_started = True
        else:
            st.error(last_err or "Tidak bisa membuka sumber video.")
            st.info("Pastikan webcam tersedia & tidak sedang dipakai aplikasi lain.")

    # Info source & tombol Stop
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(f"Source: **{st.session_state.rt_src}**  |  AUTO_START_RT: {AUTO_START_RT}")
    with col_btn:
        if st.button("⏹️ Stop", use_container_width=True) and st.session_state.rt_streaming:
            st.session_state.rt_streaming = False

    rt_image_placeholder = st.empty()
    rt_info_placeholder  = st.empty()

    # Streaming loop
    if st.session_state.rt_streaming:
        cap, err = open_video_capture(st.session_state.rt_src)
        if err:
            st.error(err)
            st.session_state.rt_streaming = False
        else:
            st.success("Streaming aktif. Gunakan tombol Stop untuk berhenti.")
            frame_count = 0
            prev_t = datetime.now()
            stride = 3  # proses 1 dari 3 frame (hemat CPU)

            while st.session_state.rt_streaming:
                ok, frame = cap.read()
                if not ok or frame is None:
                    st.warning("Frame tidak terbaca. Periksa koneksi/kamera...")
                    st.experimental_sleep(0.1)
                    continue

                frame_count += 1
                do_infer = (frame_count % stride == 0)

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if do_infer:
                    overlay_pil, _ = run_inference_on_image(
                        model, pil_frame, conf=FIXED_CONF, iou=FIXED_IOU, imgsz=FIXED_IMGSZ
                    )
                    show_pil = overlay_pil if overlay_pil is not None else pil_frame
                else:
                    show_pil = pil_frame

                now = datetime.now(); dt = (now - prev_t).total_seconds()
                fps = (1.0 / dt) if dt > 0 else 0.0
                prev_t = now

                rt_image_placeholder.image(show_pil, caption=f"Real-Time Detection (stride={stride})", use_container_width=True)
                rt_info_placeholder.info(f"FPS ~ {fps:.1f} | Source: {st.session_state.rt_src}")

                st.experimental_sleep(0.001)

            try: cap.release()
            except Exception: pass
            st.warning("Streaming dihentikan.")
    else:
        st.info("Real-Time belum aktif. (Kamera belum tersedia atau dipakai aplikasi lain)")

# ==========================
# Footer
# ==========================
st.divider()
st.caption("🔧 Car Damage Detection using YOLOv11 Instance Segmentation")
st.caption("⚠️ Automated severity assessment - verify with professional inspection")
