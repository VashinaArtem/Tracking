# Tracking

**A camera that understands what you want to follow — and keeps it in frame.**

Say or type something like *hand*, *phone*, or *red cup*. The model finds it in the video, draws a box around it, and — if servos are connected — pans and tilts the camera so the target stays centered.

Under the hood is [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V-4_5): a vision–language model that takes an image plus a short text prompt and answers *where* the object is.

---

## Why client + server

A full vision model is heavy. Running it next to the camera on a laptop is painful. This project splits the work:

| Role | Where it runs | What it does |
|------|---------------|--------------|
| **Server** | Machine with a GPU | Receives a frame + prompt, runs MiniCPM-V, returns a bounding box |
| **Client** | Laptop / mini-PC by the camera | Captures video, talks to the server, draws the box, listens for voice, drives servos |

The client stays light. The “brain” stays on the GPU.

```
  mic / keyboard              camera
        │                       │
        ▼                       ▼
   "find the hand"  ──────►   client
                                │
                         JPEG + prompt
                                │
                                ▼
                          GPU server
                          (MiniCPM-V)
                                │
                              bbox
                                │
                                ▼
                     overlay on screen
                   + servo motion (PID)
```

---

## Repository layout

This monorepo merges the former `Tracking_server` and `Tracking_client` projects.

```
tracking/
├── server/     # inference API + evaluation
└── client/     # camera app, voice, servos
```

### `server/`

| Path | Purpose |
|------|---------|
| [`server_gpu_inference/`](server/tracking_client_server/server_gpu_inference/) | **Production server** — FastAPI + MiniCPM-V on GPU |
| [`server_cpu_inference/`](server/tracking_client_server/server_cpu_inference/) | CPU / GGUF fallback via llama.cpp (slower, no GPU required) |
| [`tracking_client.py`](server/tracking_client_server/tracking_client.py) | Minimal client: camera + box only (no voice, no servos) |
| [`datasets/MeViS/`](server/datasets/MeViS/) | Dataset helpers and detection evaluation scripts |

### `client/`

| Path | Purpose |
|------|---------|
| [`Minicpm_hand_detect_and_track/`](client/Minicpm_hand_detect_and_track/) | **Main client** — camera, Whisper voice, hybrid tracker, PID → pan/tilt |
| [`check_usb_ports.py`](client/check_usb_ports.py) | Small helper to list serial / USB ports |

> **To run the system:** `server_gpu_inference` + `Minicpm_hand_detect_and_track`.

---

## How a session works

1. The server starts an API on port `8000`.
2. The client opens the camera (and optionally the microphone).
3. You set a target by voice or text: `hand`, `phone`, `person`, …
4. The client periodically sends a frame to the server.
5. The model returns the object’s coordinates.
6. Between LLM calls, a classical tracker keeps following the object — the feed stays smooth without hammering the GPU every frame.
7. If servos are present, a PID loop gently aims the camera at the box center.

No hardware? Use `--mock-servos` — video and overlay only.

---

## Quick start

### 1. Server (Linux + GPU)

```bash
cd server/tracking_client_server
python -m venv .venv && source .venv/bin/activate
pip install -r md_stuff/server_requirements.txt

python -m server_gpu_inference \
  --host 0.0.0.0 \
  --port 8000 \
  --num-gpus 1 \
  --image-size 1080 \
  --model openbmb/MiniCPM-V-4_5 \
  --enable-thinking
```

Health check:

```bash
curl http://SERVER_IP:8000/health
```

CPU-only (slower):

```bash
python server_cpu_inference/tracking_server_2_gguf.py --host 0.0.0.0 --port 8000
```

### 2. Client

```bash
cd client/Minicpm_hand_detect_and_track
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

List cameras:

```bash
python list_cameras.py
```

Software-only (no servos):

```bash
python main.py --mock-servos --camera 0 --image-size 1080 --server http://SERVER_IP:8000
```

With pan/tilt servos:

```bash
python main.py \
  --port /dev/tty.usbmodemXXXX \
  --pan-id 1 --tilt-id 2 \
  --camera 0 \
  --server http://SERVER_IP:8000
```

On Windows the serial port is usually `COM3`.

---

## Controls

### Voice or console text

| Command | Action |
|---------|--------|
| `hand`, `phone`, `cup`, `person`, … | Start tracking that object |
| `clear` / `stop` | Clear the current target |
| `quit` / `exit` | Quit |

### Video window keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear prompt |
| `s` | Server stats |

---

## Hardware (optional)

For physical camera following:

- USB camera
- 2× Feetech STS servos (pan / tilt)
- USB–Serial adapter
- Microphone — only if you want voice control

Calibrate servo ranges with `calibration.py` in the main client. Without hardware, always use `--mock-servos`.

---

## Dependencies

| Component | Needs |
|-----------|--------|
| GPU server | Python 3.9+, PyTorch, transformers, FastAPI, CUDA GPU |
| Client | Python 3.8+, OpenCV, requests; Whisper for voice; `feetech-servo-sdk` for servos |

Requirement files:

- [`server/.../server_requirements.txt`](server/tracking_client_server/md_stuff/server_requirements.txt)
- [`client/.../requirements.txt`](client/Minicpm_hand_detect_and_track/requirements.txt)

The GPU server downloads the model from Hugging Face on first run:  
[openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)

---

## Handy commands

```bash
nvtop          # live GPU monitor
nvidia-smi     # one-shot GPU snapshot

lsof -i :8000  # who holds port 8000

# tunnel if the server is behind NAT
ssh -L 8000:localhost:8000 user@server
python main.py --server http://localhost:8000 --mock-servos --camera 0
```

---

## Further reading

| Doc | About |
|-----|--------|
| [`client/.../README.md`](client/Minicpm_hand_detect_and_track/README.md) | Client CLI, PID tuning, troubleshooting |
| [`client/.../ARCHITECTURE.md`](client/Minicpm_hand_detect_and_track/ARCHITECTURE.md) | Client threads and modules |
| [`server/.../server.md`](server/tracking_client_server/md_stuff/server.md) | Server API and data flow |
| [`server/.../markdown/README.md`](server/tracking_client_server/server_gpu_inference/markdown/README.md) | GPU package module map |

---

## Not in this repo

- Virtual environments
- Model weights (downloaded separately)

---

## Licenses

Respect the licenses of MiniCPM-V, Whisper, llama.cpp, OpenCV, and any other components you use.
