# MiniCPM-V Hand Tracking System

Integrated system combining voice-controlled object detection (MiniCPM-V) with hardware servo tracking. The system uses Whisper for voice recognition, MiniCPM-V GGUF for fast object detection, and PID-controlled pan/tilt servos for real-time tracking.

## Features

- **Voice Control**: Natural language voice commands via Whisper ASR
- **Object Detection**: Fast INT4-quantized MiniCPM-V inference via llama.cpp
- **Servo Tracking**: PID-controlled pan/tilt servos for smooth object tracking
- **Real-time Display**: Live video feed with bounding boxes and status overlay
- **Mock Mode**: Test vision and tracking without servo hardware

## System Architecture

```
┌─────────────────────────────────────┐
│          LAPTOP                     │
│  ┌───────────────────────────────┐  │
│  │  Microphone → Whisper         │  │
│  │  Camera → MiniCPM Server      │  │
│  │  Bounding Box → PID → Servos  │  │
│  │  Display with Overlay         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Hardware Requirements

- USB Camera (webcam or external camera)
- 2x Feetech STS-series Servos (for pan/tilt mechanism)
- USB-to-Serial adapter for servo communication
- Microphone (for voice control)
- Computer with sufficient CPU/RAM for MiniCPM-V inference

## Software Requirements

- Python 3.8+
- Windows or Linux
- MiniCPM-V GGUF model files
- llama.cpp with llama-server compiled

## Installation

### 1. Install Python Dependencies

```bash
cd Minicpm_hand
pip install -r requirements.txt
```

### 2. Download MiniCPM-V GGUF Models

Download the quantized GGUF model files:
- Language model: `ggml-model-Q4_K_M.gguf`
- Vision projection: `mmproj-model-f16.gguf`

You can obtain these from the MiniCPM-V model repository or convert them using llama.cpp tools.

### 3. Build llama.cpp

Follow the llama.cpp build instructions to compile `llama-server`:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The `llama-server` executable will be in `llama.cpp/build/bin/`.

### 4. Configure Model Paths

Edit `hand_tracking_server.py` and update the paths at the top of the file:

```python
# Line 18-20: Update these paths
MODEL_PATH = "/path/to/your/ggml-model-Q4_K_M.gguf"
MMPROJ_PATH = "/path/to/your/mmproj-model-f16.gguf"
LLAMA_SERVER_PATH = "/path/to/your/llama-server"
```

### 5. Calibrate Servos (If Using Hardware)

If you have servo hardware, you need to calibrate the safe motion range first:

```bash
# Copy calibration tool from Andrew_camera
cp ../Andrew_camera/calibrate.py .

# Run calibration
python calibrate.py --port COM3 --pan-id 1 --tilt-id 2
```

This will create/update `calibration.json` with safe min/center/max positions for each servo.

## Usage

### Running the System

You need two terminal windows:

**Terminal 1: Start the MiniCPM-V Server**
```bash
cd Minicpm_hand
python hand_tracking_server.py --host localhost --port 8000
```

Wait for the message: `[STARTUP] llama-server is ready at http://127.0.0.1:8085`

**Terminal 2: Start the Tracking Client**

First, identify which camera is your robot camera:
```bash
python list_cameras.py
```

This will show all available cameras with previews. Camera 0 is usually the laptop webcam, camera 1+ are external/robot cameras.

With servo hardware (using robot camera):
```bash
python hand_tracking_client.py --port COM3 --camera 1
```

Without servo hardware (mock mode for testing):
```bash
python hand_tracking_client.py --mock-servos --camera 1
```

### Voice Commands

Just speak naturally! The system uses Voice Activity Detection (VAD) to automatically detect when you're speaking.

**Examples:**
- "hand" - Track your hand
- "phone" - Track your phone
- "cup" - Track a cup
- "person" - Track a person
- "clear" or "stop" - Stop tracking
- "quit" or "exit" - Exit the program

### Text Commands

You can also type commands in the console:
```
> hand
> phone
> cup
```

### Keyboard Controls (Video Window)

- `q` - Quit the program
- `c` - Clear current tracking prompt
- `s` - Show server statistics

## Configuration Options

### Client Options

```bash
python hand_tracking_client.py --help
```

Key options:
- `--server URL` - Server URL (default: http://localhost:8000)
- `--camera N` - Camera device ID (default: 1 for robot camera, use 0 for laptop webcam)
- `--port PORT` - Serial port for servos (COM3 on Windows, /dev/ttyUSB0 on Linux)
- `--fps N` - Detection rate in FPS (default: 1.5, recommended: 1.0-2.0)
- `--image-size N` - Max image dimension (default: 512, options: 384/512/768)
- `--whisper-model M` - Whisper model size (default: base, options: tiny/base/small/medium/large)
- `--no-voice` - Disable voice control, text input only
- `--mock-servos` - Run without servo hardware
- `--pan-id N` - Pan servo ID (default: 1)
- `--tilt-id N` - Tilt servo ID (default: 2)

### Server Options

```bash
python hand_tracking_server.py --help
```

Options:
- `--host HOST` - Host to bind to (default: 0.0.0.0)
- `--port PORT` - Port to bind to (default: 8000)

### Environment Variables (Advanced)

For llama-server tuning:
- `LLAMA_CTX_SIZE` - Context size (default: 4096)
- `LLAMA_THREADS` - Number of CPU threads (default: auto)
- `LLAMA_N_PREDICT` - Max tokens to generate (default: 1024)
- `LLAMA_TEMPERATURE` - Sampling temperature (default: 0.7)
- `LLAMA_SERVER_LOG` - Set to "1" to see llama-server logs

## Troubleshooting

### Server Won't Start

**Problem:** `FileNotFoundError: Language model not found`

**Solution:** Edit `hand_tracking_server.py` and update the MODEL_PATH, MMPROJ_PATH, and LLAMA_SERVER_PATH variables.

---

**Problem:** `llama-server not ready after 180s`

**Solution:** 
- Increase timeout: `export LLAMA_SERVER_STARTUP_TIMEOUT=300`
- Check if model files are corrupted
- Try reducing context size: `export LLAMA_CTX_SIZE=2048`

### Camera Issues

**Problem:** `Cannot open camera 0`

**Solution:**
- Check if camera is connected: `ls /dev/video*` (Linux)
- Try different camera ID: `--camera 1`
- Close other apps using the camera

### Servo Issues

**Problem:** `Failed to connect pan servo`

**Solution:**
- Check serial port: Windows Device Manager or `ls /dev/ttyUSB*` (Linux)
- Verify servo IDs match calibration.json
- Check servo power supply
- Verify baudrate (should be 1000000)
- Try different USB port

---

**Problem:** Servos move erratically or too fast

**Solution:**
- Adjust PID parameters in `hand_tracking_client.py`:
  - Reduce `kp` value (default: 0.15, try: 0.10)
  - Reduce `max_step_size` (default: 0.05, try: 0.03)
- Lower detection rate: `--fps 1.0`

### Voice Control Issues

**Problem:** Voice not recognized

**Solution:**
- Check microphone: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Speak clearly and pause after commands
- Try a different Whisper model: `--whisper-model small`
- Check microphone volume/gain settings

---

**Problem:** Voice control disabled automatically

**Solution:**
- Install audio dependencies: `pip install sounddevice webrtcvad faster-whisper`
- Check error message in console for specific issue

### Detection Issues

**Problem:** Object not detected

**Solution:**
- Use clearer, simpler object descriptions
- Ensure good lighting
- Object should be clearly visible in frame
- Try higher image size: `--image-size 768`
- Adjust detection rate: `--fps 1.0` (slower but more stable)

---

**Problem:** Slow inference times

**Solution:**
- Reduce image size: `--image-size 384`
- Reduce detection rate: `--fps 1.0`
- Check CPU usage - llama-server is CPU-intensive
- Consider GPU version (requires different setup)

## Performance Tuning

### Fast Mode (Lower quality, faster)
```bash
python hand_tracking_client.py --fps 1.0 --image-size 384 --whisper-model tiny
```

### Quality Mode (Higher quality, slower)
```bash
python hand_tracking_client.py --fps 1.5 --image-size 768 --whisper-model base
```

### Balanced Mode (Recommended)
```bash
python hand_tracking_client.py --fps 1.5 --image-size 512 --whisper-model base
```

## File Structure

```
Minicpm_hand/
├── hand_tracking_client.py      # Main client (camera + voice + servos)
├── hand_tracking_server.py      # MiniCPM-V GGUF server
├── feetech_servo.py             # Servo control library
├── pid_controller.py            # PID controller for smooth tracking
├── calibration.py               # Calibration data management
├── calibration.json             # Servo calibration data
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Development and Testing

### Test Without Servos

```bash
python hand_tracking_client.py --mock-servos --camera 0
```

This lets you test the vision and voice systems without servo hardware.

### Test Without Voice

```bash
python hand_tracking_client.py --no-voice --mock-servos
```

Use text input only (good for debugging).

### Check Server Health

```bash
curl http://localhost:8000/health
```

### View Server Stats

```bash
curl http://localhost:8000/stats
```

Or press `s` in the video window.

## Advanced: PID Tuning

To adjust servo tracking behavior, edit `hand_tracking_client.py`:

```python
# Line 118-128: PID controller initialization
self.pan_pid = PIDController(
    kp=0.15,        # Proportional gain (responsiveness)
    ki=0.00,        # Integral gain (steady-state error)
    kd=0.0,         # Derivative gain (damping)
    min_output=-1.0,
    max_output=1.0,
    integral_limit=0.15,
    deadband=0.003  # Ignore small errors
)
```

**Tuning tips:**
- **Too slow to respond**: Increase `kp` (0.15 → 0.20)
- **Overshoots/oscillates**: Decrease `kp` (0.15 → 0.10) or add `kd` (0.0 → 0.05)
- **Steady-state error**: Add small `ki` (0.0 → 0.01)
- **Jerky motion**: Increase `deadband` (0.003 → 0.01)

## Credits

Based on:
- MiniCPM-V by OpenBMMLab
- llama.cpp by Georgi Gerganov
- Whisper by OpenAI
- Andrew_camera face tracking system

## License

See individual component licenses.
