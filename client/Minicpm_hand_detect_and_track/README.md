# Tracking Client

Camera-side app for the Tracking monorepo: capture video, send frames to a MiniCPM-V server, draw the box, optionally follow the target with pan/tilt servos, and accept voice commands (wake word **Atlas**).

This folder is the **main client**. Detection runs on a remote/local inference server — not inside this process (except the optional local GGUF helper).

```
mic ──► VoiceController (VAD + Whisper, wake word "atlas")
              │
camera ──► capture ──► hybrid tracker ──► bbox
              │              ▲
              │         LLM re-check
              ▼              │
         ServerClient ───────┘
              │
              ▼
     PID ──► Feetech pan/tilt   (+ OpenCV overlay)
```

---

## What you need

| Piece | Required? |
|-------|-----------|
| Python 3.8+ | yes |
| Detection server on `:8000` | yes |
| USB camera | yes |
| Microphone | only for voice |
| 2× Feetech STS servos + USB–Serial | only for physical tracking |

**Recommended server** (from repo root):

```bash
cd ../../server/tracking_client_server
python -m server_gpu_inference \
  --host 0.0.0.0 --port 8000 --num-gpus 1 \
  --image-size 1080 --model openbmb/MiniCPM-V-4_5 --enable-thinking
```

Optional local CPU/GGUF server in this folder: `hand_tracking_server.py` (needs GGUF weights + `llama-server`). Prefer the GPU package above for normal use.

---

## Install

```bash
cd client/Minicpm_hand_detect_and_track
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Voice stack: `sounddevice`, `webrtcvad`, `faster-whisper`.  
Servos: `feetech-servo-sdk`.

---

## Quick start

**1. List cameras**

```bash
python list_cameras.py
```

**2. Run without hardware** (vision + overlay only)

```bash
python main.py --mock-servos --camera 0 --image-size 1080 --server http://SERVER_IP:8000
```

**3. Run with servos**

```bash
python main.py \
  --port /dev/tty.usbmodemXXXX \
  --pan-id 1 --tilt-id 2 \
  --camera 0 \
  --server http://SERVER_IP:8000
```

Windows serial port is usually `COM3`. Disable voice with `--no-voice` if needed.

---

## Voice control

Voice is **on by default**.

1. Say the wake word: **Atlas**
2. Then the command, e.g. *hand*, *phone*, *red cup*
3. Pause — Whisper finishes the phrase after ~3 s of silence

| You say | Effect |
|---------|--------|
| `Atlas` + object name | Start / change tracking target |
| `clear` / `stop` | Clear target |
| `quit` / `exit` | Quit |

You can also type the same commands in the console.

| Flag | Meaning |
|------|---------|
| `--voice` | Enable voice (default) |
| `--no-voice` | Text only |
| `--whisper-model tiny\|base\|small\|medium\|large` | ASR size (default `base`) |

Wake word is configured in `config/settings.py` → `VoiceConfig.wake_word` (default `"atlas"`).

---

## Keyboard (video window)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear prompt |
| `s` | Server stats |

---

## CLI reference

```bash
python main.py --help
```

Useful options:

| Option | Default | Notes |
|--------|---------|--------|
| `--server` | `http://localhost:8000` | Detection API |
| `--camera` | `1` | Device id (`0` = often laptop webcam) |
| `--fps` | `1.5` | How often frames go to the LLM |
| `--image-size` | `512` | `384` fast / `512` balanced / `768+` quality |
| `--quality` | `85` | JPEG quality for upload |
| `--tracker` | `KCF` | Classical tracker between LLM checks (`CSRT` = slower, stabler) |
| `--llm-check-interval` | `2.0` | Seconds between LLM re-checks |
| `--mock-servos` | off | No serial hardware |
| `--port` | mac-style USB serial | Servo adapter |
| `--pan-id` / `--tilt-id` | `1` / `2` | Feetech IDs |
| `--calibration-file` | `calibration.json` | Servo limits |
| `--camera-calibration` | none | Lens / fisheye JSON |
| `--clean-ui` | off | Bbox only, no text overlay |

---

## Servo calibration

Safe motion ranges live in `calibration.json`. Use the local helper:

```bash
python calibration.py --port COM3 --pan-id 1 --tilt-id 2
```

Without calibration / hardware, always use `--mock-servos`.

PID gains: `config/settings.py` → `PIDConfig` / `ServoConfig`.

| Symptom | Try |
|---------|-----|
| Too slow | Raise `kp` |
| Oscillates | Lower `kp`, or add a little `kd` |
| Jerky | Raise `deadband`, lower `max_step_size` |

---

## Layout

```
Minicpm_hand_detect_and_track/
├── main.py                 # CLI entry
├── hand_tracking_client.py # Orchestrator
├── hand_tracking_server.py # Optional local GGUF server
├── calibration.py          # Servo calibration helper
├── list_cameras.py
├── requirements.txt
├── config/                 # TrackingConfig, PID, voice settings
├── voice/                  # VAD + Whisper + wake word
├── network/                # HTTP client → /detect, /health, /stats
├── tracking/               # Hybrid LLM + OpenCV tracker
├── pipeline/               # Capture / processing / LLM workers
├── servo/                  # PID → Feetech
├── image/                  # Color + lens correction
├── ui/                     # Overlays
└── utils/                  # Coordinates, metrics
```

Deeper design notes: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Troubleshooting

**Cannot open camera**  
Try `--camera 0` / `1`, close other apps using the camera, run `list_cameras.py`.

**Cannot reach server**  
```bash
curl http://SERVER_IP:8000/health
```
Check firewall / VPN / that you started `server_gpu_inference`.

**Voice never starts**  
```bash
pip install sounddevice webrtcvad faster-whisper
python -c "import sounddevice as sd; print(sd.query_devices())"
```
Speak the wake word clearly; or use `--no-voice` and type commands.

**Servos fail to connect**  
Confirm port (`ls /dev/tty.*` / Device Manager), IDs, power, baudrate `1000000`. Use `--mock-servos` to isolate vision.

**Detection weak / slow**  
- Clearer short prompts (`hand`, not a long sentence)  
- Better lighting  
- `--image-size 768` for quality, `384` for speed  
- Lower `--fps` if the GPU is saturated  

---

## Suggested presets

```bash
# Fast smoke test
python main.py --mock-servos --camera 0 --fps 1.0 --image-size 384 --whisper-model tiny

# Balanced
python main.py --mock-servos --camera 0 --fps 1.5 --image-size 512 --whisper-model base

# Higher quality frames
python main.py --mock-servos --camera 0 --fps 1.5 --image-size 1080 --server http://SERVER_IP:8000
```
