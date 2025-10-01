# streamlit_app.py
"""
Aplicación Streamlit para detección y conteo de vehículos, con soporte
para archivos de video (procesamiento en backend) y webcam en tiempo real.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import av # Necesario para streamlit-webrtc

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from collections import defaultdict

# --- Asegurar imports desde src/ ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from processor import VideoProcessor  # noqa: E402

# ======================================================================
# CAMBIO: Se añadieron las clases auxiliares directamente en este archivo
# para que WebcamProcessor funcione sin errores de importación.
# ======================================================================
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Detections:
    def __init__(self, xyxy, confidence, class_id, tracker_id=None):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id

    @staticmethod
    def from_ultralytics(ultralytics_results, class_names_dict):
        xyxy = ultralytics_results.boxes.xyxy.cpu().numpy()
        confidence = ultralytics_results.boxes.conf.cpu().numpy()
        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        tracker_id = ultralytics_results.boxes.id.cpu().numpy().astype(int) if ultralytics_results.boxes.id is not None else None
        
        # Filtrar detecciones por clases de interés (carro y moto)
        vehicle_class_ids = [k for k, v in class_names_dict.items() if v in ["car", "motorcycle"]]
        mask = np.isin(class_id, vehicle_class_ids)

        return Detections(
            xyxy=xyxy[mask],
            confidence=confidence[mask],
            class_id=class_id[mask],
            tracker_id=tracker_id[mask] if tracker_id is not None else None,
        )

    @staticmethod
    def draw(frame: np.ndarray, detections: Detections) -> np.ndarray:
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            conf = detections.confidence[i]
            label = f"ID {detections.tracker_id[i]}" if detections.tracker_id is not None else f"Conf {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

class LineCounter:
    def __init__(self, start: Point, end: Point, classes: dict):
        self.line_start = start
        self.line_end = end
        self.tracker_state = {}  # Almacena la posición anterior de cada tracker_id
        self.counts = defaultdict(lambda: defaultdict(int)) # in/out counts por clase
        self.CLASS_NAMES_DICT = classes

    def set_line(self, start: Point, end: Point):
        self.line_start = start
        self.line_end = end

    def update(self, detections: Detections):
        if detections.tracker_id is None:
            return

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = xyxy
            # Centro del borde inferior de la caja
            center_x, center_y = int((x1 + x2) / 2), int(y2)

            # Si es la primera vez que vemos este tracker, guardamos su estado
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = (center_x, center_y)
                continue

            # Posición anterior
            prev_x, prev_y = self.tracker_state[tracker_id]

            # Verificar si la línea fue cruzada
            line_y = self.line_start.y # Asumiendo línea horizontal por simplicidad
            if (prev_y <= line_y < center_y) or (center_y <= line_y < prev_y):
                 # Lógica de conteo (simplificada, tu lógica original es más robusta)
                 # Aquí deberías obtener la clase del vehículo y actualizar el contador
                 pass # Implementa tu lógica de conteo aquí

            # Actualizar la posición del tracker
            self.tracker_state[tracker_id] = (center_x, center_y)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        cv2.line(frame, (self.line_start.x, self.line_start.y), (self.line_end.x, self.line_end.y), (0, 0, 255), 2)
        return frame
    
    def get_counts(self):
        # Devuelve un diccionario con los conteos
        # Ejemplo: return {"car_in": 1, "car_out": 0}
        # Esta parte necesita ser implementada completamente
        return dict(self.counts)

# ======================================================================

# (El resto del código sigue igual)
def _init_session() -> None:
    # ... tu función original va aquí ...
    ss = st.session_state
    ss.setdefault("thread", None)
    ss.setdefault("stop_event", None)
    ss.setdefault("running", False)
    ss.setdefault("last_csv", None)
    ss.setdefault("last_error", None)
    ss.setdefault("last_frame", None)
    ss.setdefault("progress", 0.0)
    ss.setdefault("stats", {
        "car_in": 0, "car_out": 0, "moto_in": 0, "moto_out": 0,
        "car_inv": 0, "moto_inv": 0, "last_update": None, "initial_loaded": False
    })
    if "frame_q" not in ss: ss.frame_q = Queue(maxsize=1)
    if "progress_q" not in ss: ss.progress_q = Queue(maxsize=8)
    if "finish_q" not in ss: ss.finish_q = Queue(maxsize=1)
    if "error_q" not in ss: ss.error_q = Queue(maxsize=8)
    if "stats_q" not in ss: ss.stats_q = Queue(maxsize=1)


_init_session()
st.set_page_config(page_title="Conteo de Vehículos", layout="wide")
st.title("🚗 Detección y Conteo de Vehículos (Streamlit)-Ocean")

with st.sidebar:
    st.header("Configuración")
    use_webcam = st.toggle("Usar webcam en tiempo real", value=False)
    
    uploaded_file = None
    if not use_webcam:
        uploaded_file = st.file_uploader( "Sube un video", type=["mp4", "avi", "mov", "mkv"])

    model = st.selectbox("Modelo YOLO", ["yolov8n.pt", "yolo11n.pt"], index=0)
    conf = st.slider("Confianza", 0.1, 0.8, 0.3, 0.05)
    
    st.divider()
    st.subheader("Lógica de Conteo")
    orientation = st.selectbox("Orientación de línea", ["vertical", "horizontal"], index=0)
    line_pos = st.slider("Posición de la línea", 0.1, 0.9, 0.5, 0.05)
    invert_dir = st.toggle("Invertir dirección (IN ↔ OUT)", value=False)

    st.divider()
    st.subheader("Inventario")
    init_car = st.number_input("Inventario inicial carros", min_value=0, value=0, step=1)
    init_moto = st.number_input("Inventario inicial motos", min_value=0, value=0, step=1)
    
    if not st.session_state.stats["initial_loaded"] or st.session_state.stats.get("init_car") != init_car:
        st.session_state.stats["car_inv"] = init_car
        st.session_state.stats["moto_inv"] = init_moto
        st.session_state.stats["initial_loaded"] = True
        st.session_state.stats["init_car"] = init_car

    st.divider()
    run_btn = st.button("▶️ Procesar Video") if not use_webcam else None
    stop_btn = st.button("⏹️ Detener Procesamiento") if st.session_state.running and not use_webcam else None

class WebcamProcessor(VideoTransformerBase):
    def __init__(self, cfg: AppConfig):
        self.config = cfg
        self.model = YOLO(cfg.model_name)
        
        # Mapeo de clases del modelo
        self.class_names_dict = {k: v for k, v in self.model.model.names.items() if v in ["car", "motorcycle"]}
        
        self.line_counter = LineCounter(
            start=Point(0, 0), end=Point(0, 0),
            classes=self.class_names_dict
        )

    def _update_line_geometry(self, frame_width, frame_height):
        if self.config.line_orientation == "vertical":
            start = Point(int(frame_width * self.config.line_position), 0)
            end = Point(int(frame_width * self.config.line_position), frame_height)
        else: # horizontal
            start = Point(0, int(frame_height * self.config.line_position))
            end = Point(frame_width, int(frame_height * self.config.line_position))
        self.line_counter.set_line(start, end)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        frame_height, frame_width, _ = img.shape
        
        if self.line_counter.line_start.x == 0 and self.line_counter.line_start.y == 0:
            self._update_line_geometry(frame_width, frame_height)
            
        results = self.model.track(
            img, persist=True, conf=self.config.conf, verbose=False
        )
        
        detections = Detections.from_ultralytics(results[0], self.class_names_dict)
        
        # Esta es la parte importante: el conteo real. 
        # Tu lógica original en VideoProcessor es más compleja y probablemente mejor.
        # Esto es una adaptación.
        self.line_counter.update(detections=detections)
        
        processed_img = img.copy()
        processed_img = Detections.draw(processed_img, detections)
        processed_img = self.line_counter.draw(processed_img)
        
        stats = self.line_counter.get_counts()
        # Actualiza el estado de la sesión para reflejar en la UI
        # (Esto puede causar problemas de concurrencia si no se maneja con cuidado)
        # st.session_state.stats["car_in"] = stats.get("car_in", 0)
        # ... etc
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def _save_uploaded_to_disk(file):
    if file is None: return None
    uploads = ROOT / "uploads"; uploads.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = uploads / f"{ts}_{file.name}"
    with open(dst, "wb") as f: f.write(file.getvalue())
    return str(dst)

col1, col2 = st.columns([3, 1])

with col1:
    if use_webcam:
        st.subheader("Video en Tiempo Real (Webcam)")
        cfg_webcam = AppConfig(
            model_name=model, conf=float(conf),
            line_orientation=orientation, line_position=float(line_pos),
            invert_direction=bool(invert_dir)
        )
        webrtc_streamer(
            key="webcam-streamer",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: WebcamProcessor(cfg=cfg_webcam),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.subheader("Procesamiento de Video Subido")
        frame_placeholder = st.empty()
        progress_placeholder = st.progress(0)

with col2:
    st.subheader("📊 Resumen de Vehículos")
    stats_placeholder = st.empty()

if run_btn and uploaded_file and not st.session_state.running:
    st.session_state.running = True
    
    for q_name in ("frame_q", "progress_q", "finish_q", "error_q", "stats_q"):
        q = getattr(st.session_state, q_name)
        while not q.empty():
            try: q.get_nowait()
            except Empty: break
    
    video_path = _save_uploaded_to_disk(uploaded_file)
    
    if video_path:
        cfg_video = AppConfig(
            model_name=model, conf=float(conf), iou=0.5, device=None,
            line_orientation=orientation, line_position=float(line_pos),
            invert_direction=bool(invert_dir), capacity_car=int(cap_car),
            capacity_moto=int(cap_moto), initial_inventory_car=int(init_car),
            initial_inventory_moto=int(init_moto), enable_csv=True, draw_hud=True
        )
        stop_event = threading.Event()
        st.session_state.stop_event = stop_event
        
        def on_frame_callback(frame):
            if not st.session_state.frame_q.full(): st.session_state.frame_q.put(frame)
        def on_progress_callback(p):
            if not st.session_state.progress_q.full(): st.session_state.progress_q.put(p)
        def on_stats_callback(stats):
            if not st.session_state.stats_q.full(): st.session_state.stats_q.put(stats)

        vp = VideoProcessor(
            video_source=video_path, config=cfg_video, stop_event=stop_event,
            on_frame=on_frame_callback, on_progress=on_progress_callback
        )
        
        thread = threading.Thread(target=vp.run, daemon=True)
        st.session_state.thread = thread
        thread.start()

if stop_btn:
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.running = False

while True:
    if st.session_state.running and not use_webcam:
        try:
            frame = st.session_state.frame_q.get(timeout=0.1)
            frame_placeholder.image(frame, channels="RGB")
        except Empty: pass
        try:
            progress = st.session_state.progress_q.get_nowait()
            progress_placeholder.progress(int(progress * 100))
        except Empty: pass
        if st.session_state.thread and not st.session_state.thread.is_alive():
            st.session_state.running = False
            st.info("Procesamiento de video finalizado.")
    
    def display_stats():
        with stats_placeholder.container():
            try:
                latest_stats = st.session_state.stats_q.get_nowait()
                st.session_state.stats.update(latest_stats)
            except Empty: pass

            car_in, car_out = st.session_state.stats.get('car_in', 0), st.session_state.stats.get('car_out', 0)
            moto_in, moto_out = st.session_state.stats.get('moto_in', 0), st.session_state.stats.get('moto_out', 0)
            car_inv = st.session_state.stats.get("init_car", 0) + car_in - car_out
            moto_inv = st.session_state.stats.get("init_moto", 0) + moto_in - moto_out

            st.markdown(f"""
            | Tipo | IN | OUT | Inventario |
            | :--: | :-: | :-: | :---: |
            | 🚗  | {car_in} | {car_out} | **{car_inv}** |
            | 🏍️  | {moto_in} | {moto_out} | **{moto_inv}** |
            """)
            if st.session_state.stats["last_update"]:
                st.caption(f"Actualizado: {st.session_state.stats['last_update'].strftime('%H:%M:%S')}")
    
    display_stats()
    time.sleep(0.1)