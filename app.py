# app.py
import os
import queue
import threading
from collections import deque, Counter
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path

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
st.set_page_config(page_title="Car Damage Detection (Streaming)", page_icon="🚗", layout="wide")
st.title("🚗 Car Damage Detection — Live Streaming with Segmentation")

WEIGHTS_FILE = "best.pt"          # taruh model di root app
CONF_THRES   = 0.10
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
    area = float((x2 - x1) * (y2 - y1))
    if area < 5000:
        return "Light"
    elif area < 15000:
        return "Medium"
    else:
        return "Heavy"

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.checkbox("Show Segmentation Masks", value=True)
    show_labels = st.checkbox("Show Class Labels", value=True)
    mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.subheader("🎥 Camera")
    cam_choice = st.radio(
        "Pilih kamera",
        ["Belakang (environment)", "Depan (user)", "Auto"],
        index=0,
        help="Di ponsel, 'Belakang' biasanya kamera utama. Di desktop, browser bisa mengabaikan preferensi ini."
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

    st.caption("💡 Tips: Safari/iOS butuh HTTPS agar kamera bisa dipakai. Jika perangkat tidak punya kamera belakang, browser akan fallback ke yang tersedia.")

# ==========================
# RESULT QUEUE (for right pane)
# ==========================
result_queue: "queue.Queue[dict]" = queue.Queue()
recent_detections = deque(maxlen=200)  # Simpan ringkas 200 deteksi terakhir

# ==========================
# VIDEO PROCESSOR
# ==========================
class YOLOSegProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]

        # Inference
        with self.lock:
            results = self.model.predict(
                source=img_bgr,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=IMG_SIZE,
                verbose=False
            )
        r = results[0]

        # Prepare overlay for masks
        overlay = img_bgr.copy()

        # Draw masks (no confidence text)
        if hasattr(r, "masks") and r.masks is not None:
            masks = r.masks.data.cpu().numpy()  # [N, Hm, Wm]
            if masks.size > 0:
                for mask, cls_id in zip(masks, r.boxes.cls.cpu().numpy().astype(int)):
                    mask_resized = cv2.resize(mask, (w, h))
                    m = mask_resized > 0.5
                    col = color_for(r.names.get(cls_id, str(cls_id)))
                    overlay[m] = (0.45 * np.array(col) + 0.55 * overlay[m]).astype(np.uint8)

        # Blend if any mask
        img_draw = cv2.addWeighted(
            overlay,
            mask_alpha if show_masks else 0,
            img_bgr,
            1 - (mask_alpha if show_masks else 0),
            0,
        )

        # Draw boxes + labels (without confidence)
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

                # Push compact result (no conf displayed on frame)
                try:
                    result_queue.put_nowait({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "class_name": cname,
                        "severity": severity_from_bbox(box),
                        "confidence": float(conf),  # disimpan untuk analitik, tidak ditulis di frame
                    })
                except queue.Full:
                    pass

        return av.VideoFrame.from_ndarray(img_draw, format="bgr24")

# ==========================
# WEBRTC STREAM
# ==========================
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

# ==========================
# RIGHT PANE: Live Stats
# ==========================
col_stream, col_stats = st.columns([2, 1])

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

        # live loop (uses Streamlit rerun on each script run)
        new_items = drain_results()
        recent_detections.extend(new_items)

        if len(recent_detections) > 0:
            df = pd.DataFrame(list(recent_detections))
            # Ringkas
            total = len(df)
            heavy = (df["severity"] == "Heavy").sum()
            cls_counts = df["class_name"].value_counts()

            st.metric("Detections (recent)", total)
            st.metric("Heavy", heavy)

            # Tabel mini (tanpa confidence display)
            show_df = df[["time", "class_name", "severity"]].tail(12)
            table_box.dataframe(show_df, use_container_width=True, height=320)

            chart_box1.bar_chart(cls_counts)
            sev_counts = df["severity"].value_counts()
            chart_box2.bar_chart(sev_counts)
        else:
            status_box.info("Menunggu deteksi…")
    else:
        status_box.warning("Streaming belum dimulai. Klik **Start** pada komponen kamera di atas.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("📸 Live YOLO segmentation — labels only (no confidence text).")
