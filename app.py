# app.py
import os
import io
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# WebRTC
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# ==========================
# APP CONFIG
# ==========================
st.set_page_config(page_title="Car Damage Detection (Streaming + Upload)", page_icon="🚗", layout="wide")
st.title("🚗 Car Damage Detection — Live Segmentation (Camera & Image Upload)")

WEIGHTS_FILE = "best.pt"          # taruh model di root app
CONF_THRES   = 0.25
IOU_THRES    = 0.7
IMG_SIZE     = 416

# ==========================
# MODEL
# ==========================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

@st.cache_resource(show_spinner=True)
def load_model():
    if not YOLO_AVAILABLE:
        return None
    if not Path(WEIGHTS_FILE).exists():
        return None
    m = YOLO(WEIGHTS_FILE)
    try:
        m.fuse()
    except Exception:
        pass
    return m

model = load_model()
if model is None:
    st.error("❌ Model tidak ditemukan/terload. Pastikan `best.pt` ada di direktori yang sama.")
    st.stop()
st.success(f"✅ Model loaded: {WEIGHTS_FILE}")

# ==========================
# UTIL
# ==========================
COLORS = {
    # sesuaikan jika kelasmu beda; default fallback biru
    "dent": (0, 114, 255),
    "scratch": (255, 159, 0),
    "crack": (255, 56, 56),
    "rust": (0, 176, 80),
    "default": (0, 153, 255),
}

def color_for(name: str):
    return COLORS.get(name.lower(), COLORS["default"])

def put_label(img, text, x, y):
    # label kecil tanpa confidence
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 8, y), color=(0, 0, 0), thickness=-1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def severity_from_bbox(box):
    x1, y1, x2, y2 = box
    area = float(max(x2 - x1, 0) * max(y2 - y1, 0))
    if area < 5000:
        return "Light"
    elif area < 15000:
        return "Medium"
    else:
        return "Heavy"

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# ==========================
# SIDEBAR (BERLAKU UNTUK KEDUA MODE)
# ==========================
with st.sidebar:
    st.header("⚙️ Settings (Global)")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.checkbox("Show Segmentation Masks", value=True)
    show_labels = st.checkbox("Show Class Labels", value=True)
    mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.subheader("🎥 Camera Pref")
    cam_choice = st.radio(
        "Pilih kamera",
        ["Belakang (environment)", "Depan (user)", "Auto"],
        index=0,
        help="Di ponsel, 'Belakang' biasanya kamera utama. Di desktop, preferensi bisa diabaikan oleh browser."
    )

    res_label = st.selectbox(
        "Resolusi video",
        ["1280×720", "1920×1080", "640×480"],
        index=0
    )
    fps = st.slider("FPS", 10, 60, 24, 1)

    # parse resolusi
    try:
        w, h = map(int, res_label.replace("×", "x").split("x"))
    except Exception:
        w, h = 1280, 720

    # Build constraints dinamis
    video_constraints = {
        "width": {"ideal": w},
        "height": {"ideal": h},
        "frameRate": {"ideal": fps, "max": min(30, fps)},  # 30 aman untuk WebRTC
    }
    if cam_choice == "Belakang (environment)":
        video_constraints["facingMode"] = {"exact": "environment"}
    elif cam_choice == "Depan (user)":
        video_constraints["facingMode"] = {"exact": "user"}
    else:
        video_constraints["facingMode"] = {"ideal": "environment"}

    st.caption("💡 Safari/iOS perlu HTTPS agar kamera aktif. Jika perangkat tidak punya kamera belakang, akan fallback otomatis.")

# ==========================
# RESULT QUEUE (untuk panel kanan Live)
# ==========================
result_queue: "queue.Queue[dict]" = queue.Queue()
recent_detections = deque(maxlen=200)  # Simpan ringkas 200 deteksi terakhir

# ==========================
# RENDERING / INFERENCE SHARED (untuk Upload & Camera)
# ==========================
def render_segmentation_on_bgr(img_bgr: np.ndarray):
    """Jalankan inference dan gambar mask/box/label sesuai setting. Return (img_draw, records)."""
    h, w = img_bgr.shape[:2]
    results = model.predict(
        source=img_bgr,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        verbose=False
    )
    r = results[0]
    overlay = img_bgr.copy()

    records = []

    # Masks
    if show_masks and hasattr(r, "masks") and r.masks is not None:
        masks = r.masks.data.cpu().numpy()  # [N, Hm, Wm]
        if masks.size > 0:
            # Resize tiap mask ke ukuran frame
            clss = r.boxes.cls.cpu().numpy().astype(int) if (hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0) else [0] * masks.shape[0]
            for mask, cls_id in zip(masks, clss):
                mask_resized = cv2.resize(mask, (w, h))
                m = mask_resized > 0.5
                cname = r.names.get(cls_id, str(cls_id))
                col = color_for(cname)
                overlay[m] = (0.45 * np.array(col) + 0.55 * overlay[m]).astype(np.uint8)

    # Blend
    img_draw = cv2.addWeighted(
        overlay,
        mask_alpha if show_masks else 0,
        img_bgr,
        1 - (mask_alpha if show_masks else 0),
        0,
    )

    # Boxes + Labels
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = r.names

        for box, cid, conf in zip(xyxy, clss, confs):
            x1, y1, x2, y2 = box.astype(int)
            cname = names.get(cid, str(cid))
            col = color_for(cname)
            if show_boxes:
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), col, 3)
            if show_labels:
                put_label(img_draw, cname, x1, y1)

            records.append({
                "class_name": cname,
                "severity": severity_from_bbox(box),
                "confidence": float(conf),  # untuk tabel/analitik (tidak ditulis di frame)
                "bbox": [int(v) for v in [x1, y1, x2, y2]],
            })

    return img_draw, records

# ==========================
# VIDEO PROCESSOR (Camera)
# ==========================
class YOLOSegProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            img_draw, records = render_segmentation_on_bgr(img_bgr)

        # push ke queue untuk statistik live
        for rec in records:
            try:
                result_queue.put_nowait({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "class_name": rec["class_name"],
                    "severity": rec["severity"],
                    "confidence": rec["confidence"],
                })
            except queue.Full:
                pass

        return av.VideoFrame.from_ndarray(img_draw, format="bgr24")

# ==========================
# TABS: Camera | Upload
# ==========================
tab_cam, tab_upload = st.tabs(["📹 Live Camera", "🖼️ Upload Image"])

# ---------- TAB CAMERA ----------
with tab_cam:
    col_stream, col_stats = st.columns([2, 1])

    with col_stream:
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_ctx = webrtc_streamer(
            key="yolo-seg-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=YOLOSegProcessor,
            media_stream_constraints={
                "video": video_constraints,
                "audio": False,
            },
        )

    with col_stats:
        st.subheader("📊 Live Results")
        status_box = st.empty()
        table_box = st.empty()
        chart_box1 = st.empty()
        chart_box2 = st.empty()

        # Drain queue periodically
        def drain_results():
            drained = []
            while True:
                try:
                    item = result_queue.get_nowait()
                    drained.append(item)
                except queue.Empty:
                    break
            return drained

        if webrtc_ctx.state.playing:
            status_box.info("🔴 Streaming aktif — arahkan kamera ke area kerusakan.")
            # live loop (Streamlit rerun)
            new_items = drain_results()
            recent_detections.extend(new_items)

            if len(recent_detections) > 0:
                df = pd.DataFrame(list(recent_detections))
                total = len(df)
                heavy = (df["severity"] == "Heavy").sum()
                cls_counts = df["class_name"].value_counts()

                st.metric("Detections (recent)", total)
                st.metric("Heavy", heavy)

                show_df = df[["time", "class_name", "severity"]].tail(12)
                table_box.dataframe(show_df, use_container_width=True, height=320)

                chart_box1.bar_chart(cls_counts)
                sev_counts = df["severity"].value_counts()
                chart_box2.bar_chart(sev_counts)
            else:
                status_box.info("Menunggu deteksi…")
        else:
            status_box.warning("Streaming belum dimulai. Klik **Start** pada komponen kamera di kiri.")

# ---------- TAB UPLOAD ----------
with tab_upload:
    st.subheader("🖼️ Upload Image untuk Deteksi")
    files = st.file_uploader(
        "Pilih 1 atau beberapa gambar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Gambar akan dianotasi dengan mask/box/label sesuai pengaturan di sidebar."
    )

    if files:
        for i, f in enumerate(files, start=1):
            try:
                pil = Image.open(f)
            except Exception as e:
                st.error(f"Gagal membuka file {f.name}: {e}")
                continue

            bgr = pil_to_bgr(pil)
            img_draw, records = render_segmentation_on_bgr(bgr)

            # Layout hasil
            st.markdown(f"**#{i}. {f.name}**")
            st.image(bgr_to_pil(img_draw), use_container_width=True)

            if records:
                df_rec = pd.DataFrame(records)[["class_name", "severity", "confidence", "bbox"]]
                st.caption("Ringkasan deteksi (confidence ditampilkan hanya di tabel):")
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.info("Tidak ada deteksi pada gambar ini.")

            # Download tombol
            buf = io.BytesIO()
            bgr_to_pil(img_draw).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"{Path(f.name).stem}_annotated.png",
                mime="image/png",
                use_container_width=False,
            )
            st.markdown("---")

# ==========================
# FOOTER
# ==========================
st.caption("📸 YOLO instance segmentation — label on-frame tanpa confidence; confidence tersedia di tabel.")