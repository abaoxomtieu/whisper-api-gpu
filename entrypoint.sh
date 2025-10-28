#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d /app/.venv ]; then
  source /app/.venv/bin/activate
fi

# Print GPU availability
python - <<'PY'
try:
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device count:', torch.cuda.device_count())
        print('CUDA device name:', torch.cuda.get_device_name(0))
except Exception as e:
    print('Torch check failed:', e)
PY

exec uvicorn app:app --host 0.0.0.0 --port 8005
