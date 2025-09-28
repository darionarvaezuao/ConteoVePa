# streamlit_app.py
"""
Aplicación Streamlit para detección y conteo de vehículos.

Características:
- Configuración de fuente de video (archivo o webcam).
- Selección de modelo YOLO y umbral de confianza.
- Definición de línea de conteo (posición, orientación, dirección).
- Configuración de capacidades y guardado CSV.
- Visualización en tiempo real de frames, progreso y errores.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import streamlit as st

# --- Asegurar imports desde src/ ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from processor import VideoProcessor  # noqa: E402


# ----------------------------------------------------------------------
# Estado de sesión
# ----------------------------------------------------------------------
def _init_session() -> None:
    """Inicializa variables de session_state usadas en la app."""
    ss = st.session_state
    ss.setdefault("thread", None)
    ss.setdefault("stop_event", None)
    ss.setdefault("running", False)

    # Estado visible
    ss.setdefault("last_csv", None)
    ss.setdefault("last_error", None)
    ss.setdefault("last_frame", None)
    ss.setdefault("progress", 0.0)
    
    # Estadísticas en tiempo real
    ss.setdefault("stats", {
        "car_in": 0,
        "car_out": 0,
        "moto_in": 0,
        "moto_out": 0,
        "car_inv": 0,
        "moto_inv": 0,
        "last_update": None,
        "initial_loaded": False
    })

    # Colas de comunicación (solo se crean una vez)
    if "frame_q" not in ss:
        ss.frame_q = Queue(maxsize=1)
    if "progress_q" not in ss:
        ss.progress_q = Queue(maxsize=8)
    if "finish_q" not in ss:
        ss.finish_q = Queue(maxsize=1)
    if "error_q" not in ss:
        ss.error_q = Queue(maxsize=8)
    if "stats_q" not in ss:
        ss.stats_q = Queue(maxsize=1)


# ----------------------------------------------------------------------
# Configuración de página
# ----------------------------------------------------------------------
_init_session()
st.set_page_config(page_title="Conteo de Vehículos", layout="centered")
st.title("🚗 Detección y Conteo de Vehículos (Streamlit) - Update CI/CD - Prueba en vivo")

# ----------------------------------------------------------------------
# Sidebar con parámetros de ejecución
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    # Fuente
    use_webcam = st.toggle("Usar webcam (ID 0)", value=False)
    uploaded = None
    if not use_webcam:
        uploaded = st.file_uploader(
            "Sube un video (mp4/avi/mov/mkv)",
            type=["mp4", "avi", "mov", "mkv"],
        )

    # Modelo y confianza
    model = st.selectbox("Modelo YOLO", ["yolo11n.pt", "yolov8n.pt", "yolo12n.pt"], index=0)
    conf = st.slider("Confianza", 0.10, 0.80, 0.30, step=0.01)

    # Línea de conteo
    orientation = st.selectbox("Orientación de línea", ["horizontal", "vertical"], index=1)
    line_pos = st.slider("Posición de la línea", 0.10, 0.90, 0.50, step=0.01)
    invert_dir = st.toggle("Invertir dirección (IN ↔ OUT)", value=False)

    # Capacidades e inventario inicial
    st.subheader("Capacidades")
    cap_car = st.number_input("Capacidad carros", min_value=0, value=50, step=1)
    cap_moto = st.number_input("Capacidad motos", min_value=0, value=50, step=1)
    
    st.subheader("Inventario Inicial")
    init_car = st.number_input("Inventario inicial carros", min_value=0, value=0, step=1)
    init_moto = st.number_input("Inventario inicial motos", min_value=0, value=0, step=1)
    
    # Actualizar stats con inventario inicial si cambia
    if not st.session_state.stats["initial_loaded"] or \
       st.session_state.stats["car_inv"] != init_car or \
       st.session_state.stats["moto_inv"] != init_moto:
        st.session_state.stats.update({
            "car_inv": init_car,
            "moto_inv": init_moto,
            "initial_loaded": True,
            "last_update": datetime.now()
        })

    # Visualización
    st.subheader("Visualización")
    draw_hud = st.toggle("Mostrar estadísticas en video", value=False,
                        help="Mostrar panel de información sobre el video (IN/OUT, inventario)")
    
    # CSV
    st.subheader("Reportes")
    enable_csv = st.toggle("Guardar CSV de eventos", value=True)
    csv_dir = st.text_input("Carpeta CSV", value=str(ROOT / "reports"))
    csv_name = st.text_input("Nombre de archivo (sin .csv)", value="")

    st.divider()
    run_btn = st.button("▶️ Procesar")
    stop_btn = st.button("⏹️ Detener")


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def _save_uploaded_to_disk(file) -> str | None:
    """Guarda archivo subido en disco y devuelve la ruta.
    Validates and saves uploaded media file, performs cleanup if needed."""
    if file is None:
        return None
        
    # Validate file size (max 500MB)
    MAX_SIZE = 500 * 1024 * 1024  # 500MB in bytes
    file_size = len(file.getvalue())
    if file_size > MAX_SIZE:
        raise ValueError(f"El archivo es demasiado grande. Máximo permitido: 500MB")
    
    # Setup uploads directory and cleanup old files
    uploads = ROOT / "uploads"
    uploads.mkdir(exist_ok=True)
    
    # Cleanup files older than 24 hours
    for old_file in uploads.glob("*.*"):
        if time.time() - old_file.stat().st_mtime > 86400:  # 24 hours
            try:
                old_file.unlink()
            except Exception:
                pass
    
    # Save new file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = uploads / f"{ts}_{file.name}"
    try:
        with open(dst, "wb") as f:
            f.write(file.getvalue())
        return str(dst)
    except Exception as e:
        raise IOError(f"Error al guardar archivo: {e}")

def update_stats_from_processor():
        try:
            if not hasattr(st.session_state, 'thread') or not st.session_state.thread:
                return
            
            vp = st.session_state.thread
            if not hasattr(vp, '_prev_counts') or not hasattr(vp, 'config'):
                return
                
            # Obtener valores actuales
            car_in = vp._prev_counts.get("car_in", 0)
            car_out = vp._prev_counts.get("car_out", 0)
            moto_in = vp._prev_counts.get("moto_in", 0)
            moto_out = vp._prev_counts.get("moto_out", 0)
            
            # Calcular inventario real
            car_inv = int(vp.config.initial_inventory_car) + car_in - car_out
            moto_inv = int(vp.config.initial_inventory_moto) + moto_in - moto_out
            
            # Actualizar estadísticas
            new_stats = {
                "car_in": car_in,
                "car_out": car_out,
                "moto_in": moto_in,
                "moto_out": moto_out,
                "car_inv": car_inv,
                "moto_inv": moto_inv,
                "last_update": datetime.now()
            }
            try:
                if st.session_state.stats_q.full():
                    st.session_state.stats_q.get_nowait()
                st.session_state.stats_q.put_nowait(new_stats)
            except Exception:
                pass  # Silently fail if queue operations fail
        except Exception as e:
            try:
                st.session_state.error_q.put_nowait(f"Error updating stats: {e}")
            except Exception:
                print(f"Error updating stats: {e}")


# ----------------------------------------------------------------------
# Botón detener
# ----------------------------------------------------------------------
if stop_btn and st.session_state.running and st.session_state.stop_event:
    st.session_state.stop_event.set()

# ----------------------------------------------------------------------
# Botón procesar
# ----------------------------------------------------------------------
if run_btn and not st.session_state.running:
    # Resetear estado visible
    st.session_state.last_error = None
    st.session_state.last_csv = None
    st.session_state.last_frame = None
    st.session_state.progress = 0.0

    # Vaciar colas
    for qname in ("frame_q", "progress_q", "finish_q", "error_q"):
        q: Queue = getattr(st.session_state, qname)
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    # Fuente
    if use_webcam:
        source: int | str = 0
    else:
        try:
            path = _save_uploaded_to_disk(uploaded)
            if not path or not os.path.exists(path):
                st.error("Debes subir un video válido o activar webcam.")
                st.stop()
            source = str(Path(path))  # Normalize path for Windows
        except (ValueError, IOError) as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Error inesperado al procesar archivo: {e}")
            st.stop()

    # Configuración
    cfg = AppConfig(
        model_name=model,
        conf=float(conf),
        iou=0.5,
        device=None,
        line_orientation=orientation,
        line_position=float(line_pos),
        invert_direction=bool(invert_dir),
        capacity_car=int(cap_car),
        capacity_moto=int(cap_moto),
        initial_inventory_car=int(init_car),
        initial_inventory_moto=int(init_moto),
        enable_csv=bool(enable_csv),
        csv_dir=csv_dir.strip() or "reports",
        csv_name=csv_name.strip(),
        draw_hud=bool(draw_hud),
    )

    stop_event = threading.Event()

    # Callbacks (definidos aquí para capturar variables por cierre)
    frame_q, progress_q, finish_q, error_q, stats_q = (
        st.session_state.frame_q,
        st.session_state.progress_q,
        st.session_state.finish_q,
        st.session_state.error_q,
        st.session_state.stats_q,
    )

    def cb_on_frame(frame_rgb: np.ndarray):
        try:
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put_nowait(frame_rgb)

            # Actualizar estadísticas en cada frame usando inventario inicial
            if hasattr(vp, "_prev_counts"):
                car_in = vp._prev_counts.get("car_in", 0)
                car_out = vp._prev_counts.get("car_out", 0)
                moto_in = vp._prev_counts.get("moto_in", 0)
                moto_out = vp._prev_counts.get("moto_out", 0)
                stats = {
                    "car_in": car_in,
                    "car_out": car_out,
                    "moto_in": moto_in,
                    "moto_out": moto_out,
                    "car_inv": int(cfg.initial_inventory_car) + car_in - car_out,
                    "moto_inv": int(cfg.initial_inventory_moto) + moto_in - moto_out,
                    "last_update": datetime.now(),
                }
                if not stats_q.full():
                    stats_q.put_nowait(stats)
        except Exception as e:
            try:
                error_q.put_nowait(f"cb_on_frame: {e}")
            except Exception:
                pass

    def cb_on_progress(p: float):
        try:
            if progress_q.full():
                progress_q.get_nowait()
            progress_q.put_nowait(float(p))
        except Exception:
            pass

    def cb_on_error(msg: str):
        try:
            error_q.put_nowait(str(msg))
        except Exception:
            pass

    def make_cb_on_finish(vp_: VideoProcessor):
        def _cb():
            info = {"csv": getattr(vp_, "_csv_path_str", None)}
            if not finish_q.empty():
                finish_q.get_nowait()
            finish_q.put_nowait(info)
        return _cb

    # Instanciar procesador
    vp = VideoProcessor(
        video_source=source,
        config=cfg,
        stop_event=stop_event,
        on_error=cb_on_error,
        on_finish=None,
        display=False,
        on_frame=cb_on_frame,
        on_progress=cb_on_progress,
    )
    vp.on_finish = make_cb_on_finish(vp)

    # Guardar en sesión y lanzar hilo
    st.session_state.thread = vp
    st.session_state.stop_event = stop_event
    st.session_state.running = True
    vp.start()


# ----------------------------------------------------------------------
# Visualización en UI
# ----------------------------------------------------------------------
# Crear dos columnas: una para el video y otra para el resumen
col1, col2 = st.columns([7, 3])

# Columna izquierda: video
with col1:
    # Video
    frame_placeholder = st.empty()
    
    # Barra de progreso compacta
    progress_placeholder = st.progress(int(st.session_state.progress * 100))

with col2:
    st.subheader("📊 Resumen de Vehículos")
    
    def update_stats():
        # Obtener valores actuales y calcular inventario
        car_in = st.session_state.stats.get('car_in', 0)
        car_out = st.session_state.stats.get('car_out', 0)
        moto_in = st.session_state.stats.get('moto_in', 0)
        moto_out = st.session_state.stats.get('moto_out', 0)
        
        # Calcular inventario real
        car_inv = init_car + car_in - car_out
        moto_inv = init_moto + moto_in - moto_out
        
        # Títulos de columnas
        cols_header = st.columns(4)
        with cols_header[0]:
            st.markdown("**Tipo**")
        with cols_header[1]:
            st.markdown("**IN**")
        with cols_header[2]:
            st.markdown("**OUT**")
        with cols_header[3]:
            st.markdown("**INV**")
            
        # Contenedor para las estadísticas
        with st.container():
            # Carros
            car_cols = st.columns(4)
            with car_cols[0]:
                st.markdown("🚗")
            with car_cols[1]:
                st.markdown(f"**{car_in}**")
            with car_cols[2]:
                st.markdown(f"**{car_out}**")
            with car_cols[3]:
                st.markdown(f"**{car_inv}**")
            
            # Motos
            moto_cols = st.columns(4)
            with moto_cols[0]:
                st.markdown("🏍️")
            with moto_cols[1]:
                st.markdown(f"**{moto_in}**")
            with moto_cols[2]:
                st.markdown(f"**{moto_out}**")
            with moto_cols[3]:
                st.markdown(f"**{moto_inv}**")
        
        if st.session_state.stats["last_update"]:
            st.caption(f"Actualizado: {st.session_state.stats['last_update'].strftime('%H:%M:%S')}")
    
    update_stats()

# Procesar actualizaciones de frame y progreso
try:
    while not st.session_state.frame_q.empty():
        st.session_state.last_frame = st.session_state.frame_q.get_nowait()
except Empty:
    pass

try:
    while not st.session_state.progress_q.empty():
        st.session_state.progress = float(st.session_state.progress_q.get_nowait())
except Empty:
    pass

# Procesar errores
try:
    while not st.session_state.error_q.empty():
        st.session_state.last_error = str(st.session_state.error_q.get_nowait())
except Empty:
    pass

# Procesar estadísticas (más reciente)
try:
    stats = None
    while not st.session_state.stats_q.empty():
        stats = st.session_state.stats_q.get_nowait()
    if stats:
        st.session_state.stats.update(stats)
        # Forzar actualización inmediata
        st.session_state.stats = dict(st.session_state.stats)
except Empty:
    pass

# Procesar finalización
try:
    info = st.session_state.finish_q.get_nowait()
    st.session_state.running = False
    st.session_state.last_csv = info.get('csv')
except Empty:
    pass

# Actualizar UI
if st.session_state.last_frame is not None:
    frame_placeholder.image(st.session_state.last_frame, channels="RGB")
    
progress_placeholder.progress(int(st.session_state.progress * 100))

if st.session_state.last_error:
    st.error(f"Error: {st.session_state.last_error}")

# Actualización del resumen ya se maneja en la columna derecha.

if st.session_state.running:
    st.info("Procesando… puedes detener con el botón de la izquierda.")
else:
    if st.session_state.thread is None:
        st.success("Listo para procesar.")
    else:
        st.info("Ejecución finalizada.")

# CSV descarga
if st.session_state.last_csv:
    csv_path = st.session_state.last_csv
    if csv_path and os.path.exists(csv_path):
        st.success(f"CSV generado: {csv_path}")
        with open(csv_path, "rb") as f:
            st.download_button(
                "Descargar CSV",
                data=f.read(),
                file_name=Path(csv_path).name,
                mime="text/csv",
            )

# Auto-refresh
if st.session_state.running:
    time.sleep(0.5)
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

st.caption("Tip: usa videos cortos para probar. Si usas webcam, cierra otras apps que estén usando la cámara.")
