# --- Base ligera con Python 3.12 ---
FROM python:3.12-slim

# Buenas prácticas y entorno Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias del sistema mínimas para OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia SOLO requirements primero para aprovechar la caché de capas
COPY requirements-web.txt .

# Instala dependencias (CPU-only para torch/torchvision)
# - --index-url apunta al repo CPU de PyTorch (evita bajar nvidia-* / CUDA)
# - --retries y --timeout ayudan contra cortes de red en CI
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --retries 10 --timeout 120 \
    -r requirements-web.txt


# Copia el resto del proyecto
COPY . .

# Expone la UI HTTP
EXPOSE 8080

# Comando de arranque de la app web
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
