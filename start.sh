#!/usr/bin/env bash
set -euo pipefail

echo "MODE_TO_RUN=${MODE_TO_RUN:-serverless}"
echo "DREAMO_PIPELINE_MODE=${DREAMO_PIPELINE_MODE:-smoke}"
echo "Python: $(python --version)"
echo "Working directory: $(pwd)"

if [ "${MODE_TO_RUN:-serverless}" = "pod" ]; then
  echo "Starting JupyterLab on port 8888..."
  echo "Jupyter token: ${JUPYTER_TOKEN:-dreamo}"

  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.root_dir=/app \
    --ServerApp.token="${JUPYTER_TOKEN:-dreamo}" \
    --ServerApp.allow_origin="*" \
    --ServerApp.disable_check_xsrf=True
else
  echo "Starting Runpod Serverless worker..."
  python -u /app/handler.py
fi
