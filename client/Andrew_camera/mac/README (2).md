# Face-Following Robot with Feetech Servos

Face-tracking robot using Feetech servos and OpenCV/face_recognition. It detects the largest face in the frame and keeps it centered with a PID controller, honoring calibrated servo limits.

## Requirements

- Python 3.8+
- Feetech STS-series servo(s) connected over serial
- USB camera / webcam
- Linux/macOS (tested), Windows likely works with correct serial port

## Installation

1) Install system dependencies (recommended for macOS):
```bash
# macOS
xcode-select --install
brew install cmake
```

2) Install Python packages:
```bash
pip install -r requirements.txt
```

3) (Optional) Landmark model (only if you plan to pass --shape-predictor):
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Calibration (mandatory)

Define safe min/center/max positions for each servo. Data is stored in `calibration.json`.
```bash
python calibrate.py --port /dev/ttyUSB0 --pan-id 1 --tilt-id 2
```
Follow the prompts to record MIN, CENTER, MAX for each joint. The tracker will refuse to move outside these limits.

## Run

Basic run:
```bash
python main.py --port /dev/ttyUSB0 --pan-id 1 --tilt-id 2 --camera 0
```

Mock run (no hardware; visualize tracking only):
```bash
python main.py --mock-servos --camera 0
```

Rotated camera and inverted screen motion (typical on 180°-mounted cameras):
```bash
python main.py --rotate-180 --invert-pan --invert-tilt
```

Tune for distant faces (more sensitive detection):
```bash
python main.py --detection-scale 0.75 --upsample 1 --max-fallback-upsamples 2 \
  --min-face-size 10 --max-face-size 3000 --show-all-detections
```

## Controls and UI

- Press `q` to quit
- Optionally press `p` to toggle an additional preprocessed window
- Overlays:
  - Green rectangle: selected (largest) face
  - Yellow rectangles: other faces (if `--show-all-detections`)
  - Red dot: face center
  - Blue dot: image center (target)
  - Cyan line: error vector (image center → face center)
  - On-screen text: PID output, normalized error, servo positions, FPS

## CLI Options (most useful)

Hardware/IO
- `--port /dev/ttyUSB0` serial device
- `--pan-id 1`, `--tilt-id 2` servo IDs
- `--camera 0` camera index
- `--mock-servos` run without servos
- `--calibration-file calibration.json` path to calibration

Vision
- `--detection-scale 0.3–1.0` resize factor before detection (default 0.5; higher is more accurate, slower)
- `--upsample 0–2` extra upsample steps (default 1)
- `--use-cnn` use CNN detector (very slow; HOG is default and fast)
- `--frame-skip N` process every Nth frame (speed vs smoothness)

Color/Preprocessing
- `--disable-color-correction` disable automatic white balance/tint fix
- `--green-tint-reduction 0.0–1.0` strength (default 0.3)
- `--white-balance-strength 0.0–1.0` strength (default 0.7)
- `--show-preprocessed` show an extra preprocessed window
- `--heavy-denoising` stronger but slower denoising

Orientation and Control
- `--rotate-180` rotate incoming camera frames 180°
- `--invert-pan`, `--invert-tilt` flip control directions (after rotation logic)

Detection Size and Multiples
- `--min-face-size`, `--max-face-size` short-side pixel limits (defaults 15, 2000)
- `--show-all-detections` draw all faces; largest (selected) is green

Adaptive Detection Fallback
- Enabled by default: retries when no face is found using larger scale (0.75, 1.0) and higher upsample
- `--no-detection-fallback` disable retries
- `--max-fallback-upsamples 0–2` cap extra upsample attempts (default 2)

## Behavior and Implementation Notes

- The tracker always selects the largest valid face each frame.
- If no face is found at the base settings, adaptive fallback increases detection scale and upsample within limits.
- Camera rotation: `--rotate-180` flips the control sign by default. Per-axis flips can be added with `--invert-pan` and `--invert-tilt` to match real-world motion to on-screen motion.
- Centering: PID outputs are applied as incremental deltas each frame, causing the face to converge precisely to the image center.
- Normalization: errors are normalized by half-width and half-height, so ±1 corresponds to edges of the frame on each axis.
- Safety: all servo targets are clamped to [-1, 1] normalized and then denormalized to calibrated raw ranges.

## File Overview

- `main.py` CLI and runtime wiring for `FaceTrackerRobot`
- `face_tracker_robot.py` camera loop, face detection, PID control, rendering
- `feetech_servo.py` Feetech servo wrapper (position mode, calibration helpers)
- `calibration.py` calibration data structures and JSON persistence
- `calibrate.py` interactive calibration tool
- `pid_controller.py` simple PID used for pan/tilt

## Troubleshooting

- No camera: ensure the correct `--camera` index and device permissions
- No servos: verify `--port` and servo IDs; ensure power and wiring
- "No face detected" far away: raise `--detection-scale` to 0.75 or 1.0, set `--upsample 1–2`, allow fallback (`--max-fallback-upsamples 2`)
- Jittery motion: reduce `self.max_step_size` in `FaceTrackerRobot` or lower PID `kp`; increase `--frame-skip`
- Too slow: lower `--detection-scale`, `--upsample 0`, disable `--use-cnn`, raise `--frame-skip`
- macOS build errors for dlib: install Xcode CLT and `brew install cmake`, then reinstall requirements

## License

MIT (or your preferred license). Update this section if needed.

