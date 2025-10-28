## Whisper GPU API Turbo

Blazing‑fast, production‑ready FastAPI wrapper around OpenAI Whisper running on GPU with the latest large‑v3‑turbo model. Ship speech translation/transcription in minutes, not weeks.

If this project helps you, please star it — it really helps! ⭐

### Why this repo
- **GPU‑accelerated**: Uses CUDA and the `large-v3-turbo` Whisper model via `openai-whisper`.
- **Turbo endpoints**: One‑shot turbo translate plus a fully customizable endpoint.
- **Word‑level timings (optional)**: Return per‑word timestamps and probabilities.
- **Docker‑first**: Production container, healthcheck, non‑root user, and Compose with NVIDIA support.

---

## Endpoints

Base URL: `http://localhost:8005`

- **POST `/translate-turbo`**
  - Purpose: Fast translate/transcribe with sensible turbo defaults.
  - Form fields:
    - `file` (required): Audio file upload (`.wav`, `.mp3`, etc.)
    - `include_word_timings` (optional, default `false`): `true|false|1|0|yes|no`
  - Response:
    - `text`: Output text
    - `response_time`: Processing time in seconds
    - `word_timings` (optional): Array of `{ word, start, end, probability }`

- **POST `/translate-custom`**
  - Purpose: Fine‑tune decoding options.
  - Form fields:
    - `file` (required): Audio file upload
    - `task` (optional, default `translate`): e.g., `translate` or `transcribe`
    - `temperature` (optional, default `0.0`): Decoding temperature
    - `include_word_timings` (optional, default `false`): `true|false|1|0|yes|no`
  - Response: Same schema as `/translate-turbo`.

---

## Quickstart (Docker, GPU)

Requirements:
- NVIDIA GPU + drivers
- Docker + NVIDIA Container Toolkit

Build and run with Compose (recommended):

```bash
docker network create shared-tts-network || true
docker compose up --build -d
# Service will listen on http://localhost:8005
```

Or run the image directly:

```bash
docker build -t whisper-api:cuda .
docker run -d \
  --name whisper-api \
  --gpus all \
  -p 8005:8005 \
  --restart unless-stopped \
  whisper-api:cuda
```

Healthcheck (inside container): `curl -fsS http://localhost:8005/` should return FastAPI docs page.

---

## Usage examples

### cURL (Turbo)
```bash
curl -X POST "http://localhost:8005/translate-turbo" \
  -F "file=@/path/to/audio.wav" \
  -F "include_word_timings=false"
```

### cURL (Custom)
```bash
curl -X POST "http://localhost:8005/translate-custom" \
  -F "file=@/path/to/audio.wav" \
  -F "task=translate" \
  -F "temperature=0.0" \
  -F "include_word_timings=true"
```

### Python (requests)
```python
import requests

url = "http://localhost:8005/translate-turbo"
files = {"file": open("/path/to/audio.wav", "rb")}
data = {"include_word_timings": "true"}
r = requests.post(url, files=files, data=data)
print(r.json())
```

Expected JSON (example):
```json
{
  "text": "Hello world",
  "response_time": 1.234,
  "word_timings": [
    { "word": "Hello", "start": 0.12, "end": 0.45, "probability": 0.99 },
    { "word": "world", "start": 0.46, "end": 0.88, "probability": 0.98 }
  ]
}
```

---

## Performance notes

- Default model: `large-v3-turbo` loaded on GPU (`device="cuda"`).
- Provides a low‑latency path for bilingual English ⇄ Vietnamese by seeding a multilingual prompt.
- Throughput and latency depend on GPU memory and audio length; prefer shorter input segments for best latency.

---

## Development

Run locally (GPU host):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

UVICORN_HOST=0.0.0.0 UVICORN_PORT=8005 \
python app.py
```

Then open `http://localhost:8005/` to see interactive docs.

---

## Container details

- Base: `nvidia/cuda:12.8.1-base-ubuntu24.04`
- Non‑root user, dedicated virtualenv, and healthcheck baked in.
- Ports: exposes `8005`.
- Compose file configures `--gpus all` via `runtime: nvidia` and reservations.

---

## API schema (models)

Response model for both endpoints:

```json
{
  "text": "string",
  "response_time": 0.0,
  "word_timings": [
    { "word": "string", "start": 0.0, "end": 0.0, "probability": 0.0 }
  ]
}
```

---

## Tips & troubleshooting

- Ensure NVIDIA drivers and the NVIDIA Container Toolkit are installed; test with `nvidia-smi` and `docker run --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`.
- If you see CPU fallback or CUDA errors, verify your host driver version matches the CUDA container base.
- For best latency, set `temperature=0.0` and keep audio < 30–60s per request.
- Word timings require extra alignment work; enable only when needed.

---

## Contributing & support

Issues and PRs are welcome. If this saved you time, please consider giving a star — it helps others find the project and motivates continued improvements.

