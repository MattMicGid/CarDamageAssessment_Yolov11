# app.py
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ==========================
# APP CONFIG
# ==========================
st.set_page_config(page_title="Car Damage Detection (Image Capture)", page_icon="🚗", layout="wide")
st.title("🚗 Car Damage Detection — Capture & Image Upload (Segmentation)")

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

def render_segmentation_on_bgr(img_bgr: np.ndarray, show_masks: bool, show_boxes: bool, show_labels: bool, mask_alpha: float):
    """
    Jalankan inference YOLO seg dan gambar mask/box/label sesuai setting.
    Return (img_draw, records:list[dict]).
    """
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
                "confidence": float(conf),  # tampilkan di tabel saja
                "bbox": [int(v) for v in [x1, y1, x2, y2]],
            })

    return img_draw, records

# ==========================
# SIDEBAR (GLOBAL SETTINGS)
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.checkbox("Show Segmentation Masks", value=True)
    show_labels = st.checkbox("Show Class Labels", value=True)
    mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.caption("📸 Gunakan kamera untuk capture satu foto, atau upload beberapa gambar sekaligus.")

# ==========================
# LAYOUT: 2 TABS (Capture | Upload)
# ==========================
tab_cap, tab_upload = st.tabs(["📷 Capture from Camera", "🖼️ Upload Images"])

# ---------- TAB: CAMERA CAPTURE ----------
with tab_cap:
    st.subheader("📷 Ambil Foto dari Kamera")
    st.caption("Di ponsel, biasanya akan default ke kamera belakang. Di desktop, browser memilih kamera yang tersedia.")
    img_file = st.camera_input("Ambil foto", help="Klik tombol kamera untuk mengambil gambar.")

    if img_file is not None:
        try:
            pil = Image.open(img_file)
            bgr = pil_to_bgr(pil)
            img_draw, records = render_segmentation_on_bgr(bgr, show_masks, show_boxes, show_labels, mask_alpha)

            st.image(bgr_to_pil(img_draw), use_container_width=True)

            if records:
                df_rec = pd.DataFrame(records)[["class_name", "severity", "confidence", "bbox"]]
                st.caption("Ringkasan deteksi (confidence hanya ditampilkan di tabel):")
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.info("Tidak ada deteksi pada foto ini.")

            # Download tombol
            buf = io.BytesIO()
            bgr_to_pil(img_draw).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Gagal memproses foto kamera: {e}")

# ---------- TAB: UPLOAD IMAGES ----------
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
            img_draw, records = render_segmentation_on_bgr(bgr, show_masks, show_boxes, show_labels, mask_alpha)

            st.markdown(f"**#{i}. {f.name}**")
            st.image(bgr_to_pil(img_draw), use_container_width=True)

            if records:
                df_rec = pd.DataFrame(records)[["class_name", "severity", "confidence", "bbox"]]
                st.caption("Ringkasan deteksi (confidence hanya ditampilkan di tabel):")
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.info("Tidak ada deteksi pada gambar ini.")

            buf = io.BytesIO()
            bgr_to_pil(img_draw).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"{Path(f.name).stem}_annotated.png",
                mime="image/png",
            )
            st.markdown("---")

# ==========================
# FOOTER
# ==========================
st.caption("🧠 YOLO instance segmentation — label on-frame tanpa confidence; confidence tersedia di tabel.")
# app.py
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ==========================
# APP CONFIG
# ==========================
st.set_page_config(page_title="Car Damage Detection (Image Capture)", page_icon="🚗", layout="wide")
st.title("🚗 Car Damage Detection — Capture & Image Upload (Segmentation)")

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

def render_segmentation_on_bgr(img_bgr: np.ndarray, show_masks: bool, show_boxes: bool, show_labels: bool, mask_alpha: float):
    """
    Jalankan inference YOLO seg dan gambar mask/box/label sesuai setting.
    Return (img_draw, records:list[dict]).
    """
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
                "confidence": float(conf),  # tampilkan di tabel saja
                "bbox": [int(v) for v in [x1, y1, x2, y2]],
            })

    return img_draw, records

# ==========================
# SIDEBAR (GLOBAL SETTINGS)
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.checkbox("Show Segmentation Masks", value=True)
    show_labels = st.checkbox("Show Class Labels", value=True)
    mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.caption("📸 Gunakan kamera untuk capture satu foto, atau upload beberapa gambar sekaligus.")

# ==========================
# LAYOUT: 2 TABS (Capture | Upload)
# ==========================
tab_cap, tab_upload = st.tabs(["📷 Capture from Camera", "🖼️ Upload Images"])

# ---------- TAB: CAMERA CAPTURE ----------
with tab_cap:
    st.subheader("📷 Ambil Foto dari Kamera")
    st.caption("Di ponsel, biasanya akan default ke kamera belakang. Di desktop, browser memilih kamera yang tersedia.")
    img_file = st.camera_input("Ambil foto", help="Klik tombol kamera untuk mengambil gambar.")

    if img_file is not None:
        try:
            pil = Image.open(img_file)
            bgr = pil_to_bgr(pil)
            img_draw, records = render_segmentation_on_bgr(bgr, show_masks, show_boxes, show_labels, mask_alpha)

            st.image(bgr_to_pil(img_draw), use_container_width=True)

            if records:
                df_rec = pd.DataFrame(records)[["class_name", "severity", "confidence", "bbox"]]
                st.caption("Ringkasan deteksi (confidence hanya ditampilkan di tabel):")
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.info("Tidak ada deteksi pada foto ini.")

            # Download tombol
            buf = io.BytesIO()
            bgr_to_pil(img_draw).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Gagal memproses foto kamera: {e}")

# ---------- TAB: UPLOAD IMAGES ----------
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
            img_draw, records = render_segmentation_on_bgr(bgr, show_masks, show_boxes, show_labels, mask_alpha)

            st.markdown(f"**#{i}. {f.name}**")
            st.image(bgr_to_pil(img_draw), use_container_width=True)

            if records:
                df_rec = pd.DataFrame(records)[["class_name", "severity", "confidence", "bbox"]]
                st.caption("Ringkasan deteksi (confidence hanya ditampilkan di tabel):")
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.info("Tidak ada deteksi pada gambar ini.")

            buf = io.BytesIO()
            bgr_to_pil(img_draw).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"{Path(f.name).stem}_annotated.png",
                mime="image/png",
            )
            st.markdown("---")

# ==========================
# FOOTER
# ==========================
st.caption("🧠 YOLO instance segmentation — label on-frame tanpa confidence; confidence tersedia di tabel.")
