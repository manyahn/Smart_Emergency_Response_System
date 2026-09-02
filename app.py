"""
Smart Accident Detection System
CPU-optimized for 16GB RAM — one model active at a time.

Object Detection backbones (user picks one):
  • YOLOv8n      — fastest on CPU  (~30ms/frame)
  • RT-DETR-l    — more accurate   (~120ms/frame on CPU)

Accident detection always uses best.pt (YOLOv8-based custom model).

FIX: Accident event counter now uses proper state-machine deduplication.
     A single physical accident spanning many frames counts as ONE event.
     The counter only increments when a NEW accident appears after a clear gap.
"""

import streamlit as st
import cv2
import tempfile
import requests
from ultralytics import YOLO, RTDETR
import time
from twilio.rest import Client
import numpy as np
from collections import defaultdict
import gc
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from database import init_db, log_accident
init_db()

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Accident Detection 🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0f18 0%, #141824 100%); color: #e8eaf0; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }
.stButton>button {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white; font-family: 'Rajdhani', sans-serif;
    font-size: 15px; font-weight: 700; letter-spacing: 1px;
    border-radius: 8px; border: none; padding: 10px 20px;
    box-shadow: 0 4px 15px rgba(255,68,68,0.3);
    width: 100%; margin-bottom: 6px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1d2e, #1f2340);
    border: 1px solid #2a2f4a; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px;
}
.metric-label { color: #8892b0; font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase; }
.metric-value { color: #e8eaf0; font-size: 26px;
    font-family: 'Rajdhani', sans-serif; font-weight: 700; }
.status-bar {
    background: #1a1d2e; border: 1px solid #2a2f4a;
    border-left: 3px solid #00b894; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; color: #8892b0; margin-top: 6px;
}
.model-pill {
    display:inline-block; padding:5px 14px; border-radius:20px;
    font-size:12px; font-weight:700; letter-spacing:.8px; margin:4px 0;
}
.info-box {
    background:#1a1d2e; border:1px solid #2a2f4a; border-radius:8px;
    padding:10px 14px; font-size:12px; color:#8892b0; margin-bottom:8px;
    line-height:1.8;
}
.legend-item {
    display:flex; align-items:center; gap:8px;
    padding:6px 10px; border-radius:6px;
    background:#1a1d2e; margin-bottom:5px;
    font-size:12px; color:#c8cee0; border:1px solid #2a2f4a;
}
.dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0c16 0%, #111422 100%);
    border-right: 1px solid #1e2235;
}
div[data-testid="stSidebar"] * { color: #c8cee0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  TWILIO — UPDATED WORKING CREDENTIALS
# ─────────────────────────────────────────────────────────────────
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_TO   = os.getenv("TWILIO_TO_NUMBER", "")
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN) if ACCOUNT_SID and AUTH_TOKEN else None


def send_alert(obj_summary: str):
    if not twilio_client:
        st.warning("Twilio credentials are not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your environment or .env file.")
        return

    try:
        twilio_client.messages.create(
            body=(
                f"🚨 ACCIDENT DETECTED!\n\n"
                f"📍 Location: Bengaluru, Karnataka\n"
                f"🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🚗 Objects Involved: {obj_summary}\n\n"
                f"⚠️ Please send emergency response team immediately!\n"
                f"🚑 Ambulance: 108 | 🚔 Police: 100"
            ),
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        st.success("📱 SMS Alert Sent Successfully!")
        print(f"✅ SMS sent at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        st.warning(f"SMS failed: {e}")
        print(f"❌ SMS failed: {e}")


# ─────────────────────────────────────────────────────────────────
#  COCO CLASSES WE CARE ABOUT
# ─────────────────────────────────────────────────────────────────
OBJECT_CLASSES = {
    0:  ("Person",       (255, 180,  50)),
    1:  ("Bicycle",      ( 50, 220, 255)),
    2:  ("Car",          ( 50, 255, 130)),
    3:  ("Motorcycle",   (255, 130,  50)),
    5:  ("Bus",          (180,  50, 255)),
    7:  ("Truck",        ( 50, 130, 255)),
}

ACCIDENT_CLASSES = ["car", "accident", "truck"]
try:
    with open("classes.txt") as f:
        lines = [l.strip() for l in f.read().strip().split("\n") if l.strip()]
        if lines:
            ACCIDENT_CLASSES = lines
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
#  CPU SPEED PRESETS
# ─────────────────────────────────────────────────────────────────
YOLO_IMGSZ   = 416
RTDETR_IMGSZ = 480
ACC_IMGSZ    = 416

YOLO_SKIP  = 2
RTDETR_SKIP = 4


# ─────────────────────────────────────────────────────────────────
#  GEOLOCATION
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_real_location():
    for url in ["http://ip-api.com/json/", "https://ipinfo.io/json"]:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                d = r.json()
                if "lat" in d:
                    return d["lat"], d["lon"], d.get("city", "Unknown"), d.get("regionName", "")
                if "loc" in d:
                    a, b = d["loc"].split(",")
                    return float(a), float(b), d.get("city", "Unknown"), d.get("region", "")
        except Exception:
            pass
    return 12.9716, 77.5946, "Bengaluru (fallback)", "Karnataka"


# ─────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolov8n():
    obj = YOLO("yolov8n.pt")
    acc = YOLO("best.pt")
    dummy = np.zeros((ACC_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    obj.predict(dummy, conf=0.5, verbose=False, imgsz=YOLO_IMGSZ)
    acc.predict(dummy, conf=0.1, verbose=False, imgsz=ACC_IMGSZ)
    return obj, acc


@st.cache_resource(show_spinner=False)
def load_rtdetr():
    obj = RTDETR("rtdetr-l.pt")
    acc = YOLO("best.pt")
    dummy = np.zeros((RTDETR_IMGSZ, RTDETR_IMGSZ, 3), dtype=np.uint8)
    try:
        obj.predict(dummy, conf=0.5, verbose=False, imgsz=RTDETR_IMGSZ)
    except Exception:
        pass
    acc.predict(dummy, conf=0.1, verbose=False, imgsz=ACC_IMGSZ)
    return obj, acc


# ─────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
def is_auto_rickshaw(x1, y1, x2, y2, cls_id):
    if cls_id not in (2, 3):
        return False
    w, h = (x2 - x1), (y2 - y1)
    return h > 0 and 0.75 <= (w / h) <= 1.15


def draw_box(img, x1, y1, x2, y2, label, conf, bgr, thickness=2):
    ih, iw = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
    txt = f"{label} {conf:.2f}"
    fs, ft = 0.52, 1
    (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    sy1 = max(y1 - th - bl - 6, 0)
    sy2 = y1
    cv2.rectangle(img, (x1, sy1), (x1 + tw + 8, sy2), bgr, -1)
    bright = 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]
    tc = (0, 0, 0) if bright > 140 else (255, 255, 255)
    cv2.putText(img, txt, (x1 + 3, sy2 - bl // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, fs, tc, ft, cv2.LINE_AA)


def draw_count_overlay(img, obj_counts):
    display_counts = {k: v for k, v in obj_counts.items() if k != "Accident"}
    if not display_counts:
        return
    ih, iw = img.shape[:2]
    items = sorted(display_counts.items())
    lh, pad, bw = 26, 10, 215
    bh = lh * len(items) + pad * 2
    ox, oy = iw - bw - 12, 12
    ov = img.copy()
    cv2.rectangle(ov, (ox, oy), (ox + bw, oy + bh), (8, 10, 28), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (ox, oy), (ox + bw, oy + bh), (45, 55, 100), 1)
    for i, (lbl, cnt) in enumerate(items):
        clr = (160, 220, 160)
        cv2.putText(img, f"{lbl}: {cnt}",
                    (ox + pad, oy + pad + (i + 1) * lh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, clr, 1, cv2.LINE_AA)


def draw_accident_banner(img):
    ih, iw = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (iw, 52), (0, 0, 150), -1)
    cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
    cv2.putText(img,
                "  !!! ACCIDENT DETECTED  -  ALERTING EMERGENCY SERVICES !!!",
                (10, 36), cv2.FONT_HERSHEY_DUPLEX, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_model_watermark(img, backbone):
    label = "YOLOv8n" if backbone == "yolo" else "RT-DETR-l"
    color = (50, 255, 130) if backbone == "yolo" else (100, 100, 255)
    ih, iw = img.shape[:2]
    cv2.putText(img, f"[{label}]",
                (8, ih - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  CORE FRAME PROCESSOR
# ─────────────────────────────────────────────────────────────────
def process_frame(frame, obj_model, acc_model, backbone,
                  obj_conf=0.35, acc_conf=0.20, debug=False):
    h, w  = frame.shape[:2]
    out   = frame.copy()
    counts = defaultdict(int)
    accident_found = False
    best_acc_conf  = None
    debug_list     = []

    obj_sz = YOLO_IMGSZ if backbone == "yolo" else RTDETR_IMGSZ

    obj_infer = cv2.resize(frame, (obj_sz, obj_sz))
    sx_o = w / obj_sz
    sy_o = h / obj_sz

    acc_infer = cv2.resize(frame, (ACC_IMGSZ, ACC_IMGSZ))
    sx_a = w / ACC_IMGSZ
    sy_a = h / ACC_IMGSZ

    # ── PASS 1 : Object / vehicle detection ───────────────────────
    try:
        res = obj_model.predict(obj_infer, conf=obj_conf,
                                verbose=False, imgsz=obj_sz)
        bxs = res[0].boxes if (res and res[0].boxes is not None) else []
        for b in bxs:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            x1 = int(bx1 * sx_o); y1 = int(by1 * sy_o)
            x2 = int(bx2 * sx_o); y2 = int(by2 * sy_o)
            cls  = int(b.cls[0])
            conf = float(b.conf[0])
            if cls not in OBJECT_CLASSES:
                continue
            label, color = OBJECT_CLASSES[cls]
            if is_auto_rickshaw(x1, y1, x2, y2, cls):
                label, color = "Auto Rickshaw", (0, 200, 255)
            counts[label] += 1
            draw_box(out, x1, y1, x2, y2, label, conf, color, thickness=2)
    except Exception as e:
        cv2.putText(out, f"ObjErr:{e}", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    # ── PASS 2 : Accident detection (best.pt) ─────────────────────
    try:
        a_res = acc_model.predict(acc_infer, conf=0.01,
                                  verbose=False, imgsz=ACC_IMGSZ)
        a_bxs = a_res[0].boxes if (a_res and a_res[0].boxes is not None) else []
        for b in a_bxs:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            x1 = int(bx1 * sx_a); y1 = int(by1 * sy_a)
            x2 = int(bx2 * sx_a); y2 = int(by2 * sy_a)
            cls  = int(b.cls[0])
            conf = float(b.conf[0])
            name = ACCIDENT_CLASSES[cls] if cls < len(ACCIDENT_CLASSES) else f"cls{cls}"
            debug_list.append((name, round(conf, 3)))
            is_acc = "accident" in name.lower()

            if debug and not is_acc and conf >= 0.05:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 130, 255), 1)
                cv2.putText(out, f"[{name} {conf:.2f}]",
                            (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)

            if is_acc and conf >= acc_conf:
                accident_found = True
                best_acc_conf  = conf if best_acc_conf is None else max(best_acc_conf, conf)
                counts["Accident"] += 1
                cv2.rectangle(out, (x1 - 7, y1 - 7), (x2 + 7, y2 + 7), (0, 0, 255), 6)
                cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 180), 3)
                draw_box(out, x1, y1, x2, y2, "ACCIDENT", conf, (0, 0, 255), thickness=3)
    except Exception as e:
        cv2.putText(out, f"AccErr:{e}", (8, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    draw_count_overlay(out, counts)
    if accident_found:
        draw_accident_banner(out)
    draw_model_watermark(out, backbone)

    return out, accident_found, best_acc_conf, dict(counts), debug_list


# ─────────────────────────────────────────────────────────────────
#  ACCIDENT EVENT STATE MACHINE
# ─────────────────────────────────────────────────────────────────
class AccidentEventTracker:
    def __init__(self, clear_frames: int = 10):
        self.clear_frames   = clear_frames
        self._active        = False
        self._frames_clear  = 0

    def update(self, accident_detected: bool):
        if accident_detected:
            self._frames_clear = 0
            if not self._active:
                self._active = True
                return True, True
            return False, True
        else:
            if self._active:
                self._frames_clear += 1
                if self._frames_clear >= self.clear_frames:
                    self._active       = False
                    self._frames_clear = 0
                else:
                    return False, True
            return False, False

    def reset(self):
        self._active       = False
        self._frames_clear = 0


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
for k, v in [("stop_camera", False), ("camera_running", False),
             ("active_backbone", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
#  STAT PANEL PLACEHOLDERS
# ─────────────────────────────────────────────────────────────────
frames_ph = accidents_ph = conf_ph = objects_ph = status_ph = location_ph = None


def render_stats(frames, accidents, obj_counts, status, conf=None):
    frames_ph.markdown(
        f"<div class='metric-card'><div class='metric-label'>Frames Processed</div>"
        f"<div class='metric-value'>{frames:,}</div></div>", unsafe_allow_html=True)

    col = "#ff4444" if accidents > 0 else "#00b894"
    accidents_ph.markdown(
        f"<div class='metric-card'><div class='metric-label'>Accident Events</div>"
        f"<div class='metric-value' style='color:{col}'>{accidents}</div></div>",
        unsafe_allow_html=True)

    conf_ph.markdown(
        f"<div class='metric-card'><div class='metric-label'>Last Accident Confidence</div>"
        f"<div class='metric-value'>{f'{conf:.1%}' if conf else '—'}</div></div>",
        unsafe_allow_html=True)

    display_obj = {k: v for k, v in obj_counts.items() if k != "Accident"}
    if display_obj:
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:4px 8px;border-bottom:1px solid #2a2f4a'>"
            f"<span style='font-size:13px;color:#64b5f6'>{k}</span>"
            f"<b style='font-size:14px;color:#e8eaf0'>{v}</b></div>"
            for k, v in sorted(display_obj.items()))
        objects_ph.markdown(
            f"<div class='metric-card'><div class='metric-label'>Objects Detected</div>"
            f"<div style='margin-top:6px'>{rows}</div></div>", unsafe_allow_html=True)
    else:
        objects_ph.markdown(
            "<div class='metric-card'><div class='metric-label'>Objects Detected</div>"
            "<div style='color:#8892b0;font-size:13px;margin-top:6px'>None detected</div></div>",
            unsafe_allow_html=True)

    status_ph.markdown(f"<div class='status-bar'>{status}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#e8eaf0'>🚦 SMART ACCIDENT DETECTION SYSTEM</h1>",
    unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("### 🤖 CHOOSE MODEL")
    backbone_label = st.radio(
        "Object Detection Backbone",
        ["⚡ YOLOv8n  —  Fast",
         "🔬 RT-DETR-l  —  Accurate"],
        index=0)
    backbone = "yolo" if "YOLOv8n" in backbone_label else "rtdetr"

    if st.session_state.active_backbone != backbone:
        st.session_state.active_backbone = backbone
        gc.collect()

    if backbone == "yolo":
        with st.spinner("⚙️ Loading YOLOv8n…"):
            obj_model, acc_model = load_yolov8n()
        badge = ("<span class='model-pill' style='background:#0d1f0d;"
                 "color:#32ff82;border:1px solid #32ff82'>⚡ YOLOv8n ACTIVE</span>")
        model_name  = "YOLOv8n"
        model_color = "#32ff82"
    else:
        with st.spinner("⚙️ Loading RT-DETR-l (downloads ~136 MB first time)…"):
            obj_model, acc_model = load_rtdetr()
        badge = ("<span class='model-pill' style='background:#0d0d22;"
                 "color:#8888ff;border:1px solid #8888ff'>🔬 RT-DETR-l ACTIVE</span>")
        model_name  = "RT-DETR-l"
        model_color = "#8888ff"

    st.markdown(badge, unsafe_allow_html=True)


    st.markdown("---")

    st.markdown("### 📂 INPUT SOURCE")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    run_detection = st.button("▶ Analyze Video")
    st.markdown("---")

    st.markdown("### 📡 LIVE CAMERA")
    cam_index = st.selectbox("Camera Index", [0, 1, 2, 3], index=0,
                              help="Try 0 first. If no feed, try 1 or 2.")
    go_live  = st.button("🎥 Start Live Detection")
    stop_btn = st.button("⏹ Stop Camera")
    st.markdown("---")

    # Default threshold values (sliders removed)
    obj_conf_thresh = 0.30 if backbone == "yolo" else 0.35
    acc_conf_thresh = 0.45
    clear_frames_n  = 10
    debug_mode      = False

    debug_ph = st.empty()
    st.markdown(
        "<div style='background:#1a2340;border:1px solid #2a3f6a;border-radius:8px;"
        "padding:10px;text-align:center;font-size:12px;color:#64b5f6'>"
        "📱 Twilio SMS: <b style='color:#00e676'>ACTIVE</b></div>",
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────
col_vid, col_stats = st.columns([2, 1])
stframe = col_vid.empty()

with col_stats:
    st.markdown("### 📊 LIVE STATS")
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Active Model</div>"
        f"<div class='metric-value' style='color:{model_color};font-size:22px'>"
        f"{model_name}</div>"
        f"<div style='color:#8892b0;font-size:11px;margin-top:2px'>"
        f"Accident: best.pt | CPU optimised</div></div>",
        unsafe_allow_html=True)
    frames_ph    = st.empty()
    accidents_ph = st.empty()
    conf_ph      = st.empty()
    objects_ph   = st.empty()
    status_ph    = st.empty()
    location_ph  = st.empty()

if stop_btn:
    st.session_state.stop_camera   = True
    st.session_state.camera_running = False

# ─────────────────────────────────────────────────────────────────
#  VIDEO FILE DETECTION
# ─────────────────────────────────────────────────────────────────
if uploaded_file and run_detection:
    st.session_state.stop_camera = False

    with st.spinner("📍 Locating…"):
        lat, lon, city, region = get_real_location()

    location_ph.markdown(
        f"<div class='metric-card'><div class='metric-label'>📍 Location</div>"
        f"<div style='color:#64b5f6;font-size:13px;margin-top:4px'>{city}, {region}</div>"
        f"<div style='color:#8892b0;font-size:11px'>{lat:.4f}, {lon:.4f}</div></div>",
        unsafe_allow_html=True)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 960)
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
    scale  = min(960 / vid_w, 1.0)
    dw, dh = int(vid_w * scale), int(vid_h * scale)

    skip_n = 2

    frame_count = acc_count = 0
    last_conf   = None; last_counts = {}; last_frame = None
    t_start     = time.time()

    tracker = AccidentEventTracker(clear_frames=clear_frames_n)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % skip_n != 0:
            if last_frame is not None:
                stframe.image(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
            continue

        frame = cv2.resize(frame, (dw, dh))
        frame, accident, conf, counts, _ = process_frame(
            frame, obj_model, acc_model, backbone,
            obj_conf_thresh, acc_conf_thresh, debug_mode)
        last_frame  = frame.copy()
        last_counts = counts if counts else last_counts

        is_new_event, is_active = tracker.update(accident)

        if is_new_event:
            acc_count += 1
            last_conf  = conf
            send_alert(", ".join(k for k in counts if k != "Accident") or "Unknown")
            log_accident(source=uploaded_file.name, label="Accident",
                         confidence=conf, lat=lat, lon=lon)

        now = time.time()
        fps_display = frame_count / max(now - t_start, 0.001)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)
        render_stats(frame_count, acc_count, last_counts,
                     f"🔄 Processing [{model_name}] — {fps_display:.1f} fps", last_conf)

    cap.release()
    render_stats(frame_count, acc_count, last_counts, "✅ Video complete!", last_conf)
    st.success(f"✅ Done [{model_name}] — {acc_count} accident event(s) detected.")

# ─────────────────────────────────────────────────────────────────
#  LIVE WEBCAM DETECTION
# ─────────────────────────────────────────────────────────────────
elif go_live:
    st.session_state.stop_camera   = False
    st.session_state.camera_running = True

    with st.spinner("📍 Locating…"):
        lat, lon, city, region = get_real_location()

    location_ph.markdown(
        f"<div class='metric-card'><div class='metric-label'>📍 Your Location</div>"
        f"<div style='color:#64b5f6;font-size:13px;margin-top:4px'>{city}, {region}</div>"
        f"<div style='color:#8892b0;font-size:11px'>{lat:.4f}, {lon:.4f}</div></div>",
        unsafe_allow_html=True)

    cap = None
    for idx, bk in [
        (cam_index, cv2.CAP_DSHOW),
        (cam_index, cv2.CAP_MSMF),
        (cam_index, cv2.CAP_V4L2),
        (cam_index, cv2.CAP_ANY),
    ]:
        try:
            c = cv2.VideoCapture(idx, bk)
            if c.isOpened():
                ok, f = c.read()
                if ok and f is not None and f.size > 0:
                    cap = c; break
            c.release()
        except Exception:
            pass

    if cap is None:
        for idx in range(5):
            try:
                c = cv2.VideoCapture(idx)
                if c.isOpened():
                    ok, f = c.read()
                    if ok and f is not None and f.size > 0:
                        cap = c; break
                c.release()
            except Exception:
                pass

    if cap is None:
        st.error("❌ No working webcam found. Check camera index.")
        st.session_state.camera_running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    INFER_EVERY = YOLO_SKIP if backbone == "yolo" else RTDETR_SKIP

    col_vid.markdown(
        f"<div style='color:{model_color};font-size:13px;text-align:center;margin-bottom:4px'>"
        f"🟢 LIVE [{model_name}] — {cam_w}×{cam_h} — "
        f"Inference every {INFER_EVERY} frames — Press ⏹ to stop</div>",
        unsafe_allow_html=True)

    frame_count    = 0;  acc_count    = 0
    last_conf      = None
    last_counts    = {}
    last_annotated = None
    miss_streak    = 0
    t_start        = time.time()
    fps_ema        = 0.0

    tracker = AccidentEventTracker(clear_frames=clear_frames_n)

    while not st.session_state.stop_camera:
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret or frame is None or frame.size == 0:
            miss_streak += 1
            if miss_streak > 50:
                st.error("❌ Camera feed lost. Refresh and try again.")
                break
            time.sleep(0.02)
            continue

        miss_streak  = 0
        frame_count += 1
        t_now        = time.time()
        elapsed      = max(t_now - t_start, 0.001)
        fps_ema      = 0.9 * fps_ema + 0.1 * (frame_count / elapsed)

        if frame_count % INFER_EVERY == 0:
            annotated, accident, conf, counts, dbg = process_frame(
                frame, obj_model, acc_model, backbone,
                obj_conf_thresh, acc_conf_thresh, debug_mode)

            last_annotated = annotated
            last_counts    = counts if counts else last_counts

            if debug_mode:
                if dbg:
                    rows_html = "".join(
                        f"<div style='font-size:11px;padding:1px 0;"
                        f"color:{'#ff7777' if 'accident' in n.lower() else '#ffb347'}'>"
                        f"{n} : <b>{c}</b></div>"
                        for n, c in dbg)
                    debug_ph.markdown(
                        "<div style='background:#1a1d2e;border:1px solid #ff8c00;"
                        "border-radius:6px;padding:8px'>"
                        "<b style='color:#ffb347;font-size:12px'>best.pt detections:</b>"
                        f"{rows_html}</div>", unsafe_allow_html=True)
                else:
                    debug_ph.markdown(
                        "<div style='background:#1a1d2e;border:1px solid #333;"
                        "border-radius:6px;padding:8px;font-size:11px;color:#555'>"
                        "best.pt: no detections this frame</div>", unsafe_allow_html=True)

            is_new_event, is_active = tracker.update(accident)

            if is_new_event:
                acc_count += 1
                last_conf  = conf
                send_alert(", ".join(k for k in counts if k != "Accident") or "Unknown")
                log_accident(source="Live Camera", label="Accident",
                             confidence=conf, lat=lat, lon=lon)

            status = ("🔴 LIVE — ⚠ ACCIDENT DETECTED!"
                      if is_active
                      else f"🟢 LIVE [{model_name}] — {fps_ema:.1f} fps — Monitoring…")
            render_stats(frame_count, acc_count, last_counts, status, last_conf)

        show = last_annotated if last_annotated is not None else frame
        stframe.image(cv2.cvtColor(show, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)

    cap.release()
    st.session_state.camera_running = False
    render_stats(frame_count, acc_count, last_counts, "⏹ Session ended", last_conf)
    st.info("Session ended. Press 'Start Live Detection' to begin again.")

# ─────────────────────────────────────────────────────────────────
#  IDLE STATE
# ─────────────────────────────────────────────────────────────────
else:
    with col_vid:
        st.markdown(f"""
        <div style='background:#1a1d2e;border:2px dashed #2a2f4a;border-radius:14px;
            padding:70px 40px;text-align:center;color:#8892b0'>
            <div style='font-size:56px;margin-bottom:14px'>🎥</div>
            <div style='font-family:Rajdhani,sans-serif;font-size:26px;
                color:#c8cee0;margin-bottom:10px'>Ready</div>
            <div style='margin-bottom:16px'>
                <span style='background:#141824;border:1px solid {model_color};
                    color:{model_color};border-radius:16px;padding:4px 16px;
                    font-size:13px;font-weight:700'>✅ {model_name} SELECTED</span>
            </div>

        </div>""", unsafe_allow_html=True)
    render_stats(0, 0, {}, f"Idle — {model_name} ready")