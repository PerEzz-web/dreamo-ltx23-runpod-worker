FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pod = starts JupyterLab
# serverless = starts Runpod worker
ENV MODE_TO_RUN=serverless

# Production should use Runpod cached Hugging Face model.
ENV USE_RUNPOD_CACHE=1
ENV ALLOW_HF_DOWNLOAD=0

# Dreamo fixed video settings.
ENV DEFAULT_WIDTH=832
ENV DEFAULT_HEIGHT=480
ENV LTX_WIDTH=832
ENV LTX_HEIGHT=512
ENV DEFAULT_FPS=24
ENV DEFAULT_NUM_FRAMES=73
ENV MIN_FRAMES=49
ENV MAX_FRAMES=97

# Runpod cached model location.
ENV HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# smoke = test signed URL flow without real LTX generation
# native = real Dreamo LTX generation, after developer wires it
ENV DREAMO_PIPELINE_MODE=smoke

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY dreamo_ltx_pipeline.py /app/dreamo_ltx_pipeline.py
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8888

CMD ["/app/start.sh"]
