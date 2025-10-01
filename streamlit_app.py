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

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# --- Asegurar imports desde src/ ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from processor import VideoProcessor  # noqa: E402
# Asumimos que podemos importar estas clases de tu lógica interna
from utils.detection import Detections
from utils.counting import LineCounter, Point

# ----------------------------------------------------------------------
# Estado de sesión
# ----------------------------------------------------------------------
def _init_session() -> None:
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


# ----------------------------------------------------------------------
# Configuración de página
# ----------------------------------------------------------------------
_init_session()
st.set_page_config(page_title="Conteo de Vehículos", layout="wide")
st.title("🚗 Detección y Conteo de Vehículos (Streamlit)-Ocean")

# ----------------------------------------------------------------------
# Sidebar (Tu código original)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Clase para el procesamiento de video en tiempo real con streamlit-webrtc
# ----------------------------------------------------------------------
class WebcamProcessor(VideoTransformerBase):
    def __init__(self, cfg: AppConfig):
        self.config = cfg
        self.model = YOLO(cfg.model_name)
        self.line_counter = LineCounter(
            start=Point(0, 0), end=Point(0, 0), # Se actualizará en el primer frame
            classes=cfg.CLASS_NAMES_DICT
        )
        self.class_names = self.model.model.names

    def _update_line_geometry(self, frame_width, frame_height):
        """ Actualiza la línea de conteo basado en las dimensiones del frame. """
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
        
        # Actualiza la geometría de la línea en el primer frame o si cambia el tamaño
        if self.line_counter.line_start.x == 0 and self.line_counter.line_start.y == 0:
            self._update_line_geometry(frame_width, frame_height)
            
        results = self.model.track(
            img, persist=True, conf=self.config.conf, verbose=False
        )
        
        detections = Detections.from_ultralytics(results[0], self.class_names)
        
        self.line_counter.update(detections=detections)
        
        # Dibuja los resultados
        processed_img = img.copy()
        processed_img = Detections.draw(processed_img, detections)
        processed_img = self.line_counter.draw(processed_img)
        
        # Actualiza las estadísticas en el estado de la sesión
        stats = self.line_counter.get_counts()
        st.session_state.stats["car_in"] = stats.get("car_in", 0)
        st.session_state.stats["car_out"] = stats.get("car_out", 0)
        st.session_state.stats["moto_in"] = stats.get("moto_in", 0)
        st.session_state.stats["moto_out"] = stats.get("moto_out", 0)
        st.session_state.stats["last_update"] = datetime.now()
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# (Aquí irían tus funciones auxiliares originales como _save_uploaded_to_disk, etc.)
def _save_uploaded_to_disk(file):
    # ... tu función original va aquí ...
    if file is None: return None
    uploads = ROOT / "uploads"; uploads.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = uploads / f"{ts}_{file.name}"
    with open(dst, "wb") as f: f.write(file.getvalue())
    return str(dst)
# ----------------------------------------------------------------------
# Lógica principal
# ----------------------------------------------------------------------
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

# Lógica para procesar un ARCHIVO DE VIDEO (tu código original, casi sin cambios)
if run_btn and uploaded_file and not st.session_state.running:
    st.session_state.running = True
    
    # Resetear colas y estado
    for q_name in ("frame_q", "progress_q", "finish_q", "error_q", "stats_q"):
        q = getattr(st.session_state, q_name)
        while not q.empty():
            try: q.get_nowait()
            except Empty: break
    
    video_path = _save_uploaded_to_disk(uploaded_file)
    
    if video_path:
        # Configuración completa para VideoProcessor
        cfg_video = AppConfig(
            model_name=model, conf=float(conf), iou=0.5, device=None,
            line_orientation=orientation, line_position=float(line_pos),
            invert_direction=bool(invert_dir), capacity_car=int(cap_car),
            capacity_moto=int(cap_moto), initial_inventory_car=int(init_car),
            initial_inventory_moto=int(init_moto), enable_csv=True, draw_hud=True
        )
        stop_event = threading.Event()
        st.session_state.stop_event = stop_event
        
        # Callbacks para actualizar la UI desde el hilo
        def on_frame_callback(frame):
            if not st.session_state.frame_q.full(): st.session_state.frame_q.put(frame)
        def on_progress_callback(p):
            if not st.session_state.progress_q.full(): st.session_state.progress_q.put(p)
        def on_stats_callback(stats):
            if not st.session_state.stats_q.full(): st.session_state.stats_q.put(stats)

        # Instancia y ejecución de tu VideoProcessor original
        vp = VideoProcessor(
            video_source=video_path, config=cfg_video, stop_event=stop_event,
            on_frame=on_frame_callback, on_progress=on_progress_callback
            # Nota: Asumo que tu VP llama a una cola de stats o la puedes añadir.
        )
        
        thread = threading.Thread(target=vp.run, daemon=True)
        st.session_state.thread = thread
        thread.start()

if stop_btn:
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
        st.session_state.running = False

# Bucle de actualización de la UI
while True:
    # Lógica para el modo de archivo de video
    if st.session_state.running and not use_webcam:
        try:
            frame = st.session_state.frame_q.get(timeout=0.1)
            frame_placeholder.image(frame, channels="RGB")
        except Empty:
            pass

        try:
            progress = st.session_state.progress_q.get_nowait()
            progress_placeholder.progress(int(progress * 100))
        except Empty:
            pass

        if st.session_state.thread and not st.session_state.thread.is_alive():
            st.session_state.running = False

    # Lógica para actualizar el panel de estadísticas (funciona en ambos modos)
    def display_stats():
        with stats_placeholder.container():
            # Actualizar stats desde la cola (si hay algo nuevo del VideoProcessor)
            try:
                latest_stats = st.session_state.stats_q.get_nowait()
                st.session_state.stats.update(latest_stats)
            except Empty:
                pass

            car_in = st.session_state.stats.get('car_in', 0)
            car_out = st.session_state.stats.get('car_out', 0)
            moto_in = st.session_state.stats.get('moto_in', 0)
            moto_out = st.session_state.stats.get('moto_out', 0)
            
            # Usa el inventario inicial del estado de la sesión
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
    
    # Pausa para dar tiempo a la UI a refrescarse
    time.sleep(0.1)