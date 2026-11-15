# Архитектура Hand Tracking Client

## Диаграмма компонентов

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│                    (Entry Point + CLI)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HandTrackingClient                             │
│                     (Orchestrator)                              │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Config    │  │   Metrics    │  │   Locks      │          │
│  └─────────────┘  └──────────────┘  └──────────────┘          │
└─────┬───────┬────────┬────────┬────────┬────────┬──────────────┘
      │       │        │        │        │        │
      ▼       ▼        ▼        ▼        ▼        ▼
   ┌────┐ ┌─────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
   │Net │ │Voice│ │Servo │ │Track │ │Image │ │  UI  │
   │work│ │     │ │      │ │      │ │      │ │      │
   └────┘ └─────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

## Потоки данных

### 1. Инициализация

```
main.py
  │
  ├─> TrackingConfig (from CLI args)
  │
  └─> HandTrackingClient.__init__()
        │
        ├─> ServerClient (network)
        ├─> ServoController (servo)
        ├─> HybridTracker (tracking)
        ├─> VoiceController (voice)
        ├─> ColorCorrector (image)
        └─> Visualizer (ui)
```

### 2. Основной цикл (Main Loop)

```
HandTrackingClient.run()
  │
  ├─> Camera: capture frame
  │     │
  │     └─> ColorCorrector: apply corrections
  │
  ├─> HybridTracker: track(frame, prompt)
  │     │
  │     ├─> Should use LLM?
  │     │     ├─ YES ─> ServerClient: detect(frame, prompt)
  │     │     │            └─> LLM detection result (bbox)
  │     │     │
  │     │     └─ NO ──> TrackerManager: update(frame)
  │     │                 └─> Classical tracker result (bbox)
  │     │
  │     └─> Return bbox
  │
  ├─> bbox found?
  │     │
  │     └─ YES ─> ServoController: update_position(object_center)
  │                  └─> PID control -> move servos
  │
  ├─> Visualizer: draw_bbox(frame, bbox)
  ├─> Visualizer: draw_info_overlay(frame, stats)
  │
  └─> cv2.imshow(frame)
```

### 3. Голосовое управление (параллельный поток)

```
VoiceController (background thread)
  │
  ├─> Microphone: capture audio
  │
  ├─> VAD: detect speech
  │     │
  │     └─ Speech detected ─> Buffer audio
  │
  ├─> Silence detected ─> Whisper: transcribe(audio_buffer)
  │     │
  │     └─> Transcription result
  │
  └─> on_command_callback(transcription)
        │
        └─> HandTrackingClient: update prompt / quit / clear
```

### 4. Консольный ввод (параллельный поток)

```
HandTrackingClient._prompt_input_loop() (background thread)
  │
  ├─> input() - wait for user input
  │
  └─> Update current_prompt (thread-safe with lock)
```

## Зависимости модулей

```
config/
  └─> [NO DEPENDENCIES] (pure dataclasses)

utils/
  └─> [MINIMAL DEPENDENCIES] (numpy, logging)

network/
  ├─> requests
  ├─> cv2 (OpenCV)
  └─> PIL (Pillow)

image/
  ├─> cv2 (OpenCV)
  └─> numpy

servo/
  ├─> feetech_servo (external)
  ├─> calibration (external)
  ├─> pid_controller (external)
  └─> numpy

tracking/
  ├─> cv2 (OpenCV)
  ├─> utils.coordinates
  └─> network.server_client

voice/
  ├─> sounddevice
  ├─> webrtcvad
  ├─> faster_whisper
  └─> numpy

ui/
  ├─> cv2 (OpenCV)
  └─> numpy

hand_tracking_client/
  ├─> ALL MODULES ABOVE
  └─> threading
```

## Потоки выполнения

```
┌──────────────────────────────────────────────────────────────┐
│                       Main Thread                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HandTrackingClient.run()                              │  │
│  │    - Camera capture                                    │  │
│  │    - Frame processing                                  │  │
│  │    - Tracking logic                                    │  │
│  │    - Servo control                                     │  │
│  │    - Display update                                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Voice Thread (daemon)                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  VoiceController._voice_input_loop()                   │  │
│  │    - Audio capture (sounddevice callback)             │  │
│  │    - VAD processing                                    │  │
│  │    - Whisper transcription (spawns sub-threads)        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Console Input Thread (daemon)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HandTrackingClient._prompt_input_loop()               │  │
│  │    - Blocking input() call                             │  │
│  │    - Update prompt (with lock)                         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Синхронизация

### Locks

```python
self.prompt_lock = threading.Lock()
```

**Защищает:**
- `self.current_prompt` (read/write из разных потоков)

**Используется в:**
- Main thread (read в run(), write при нажатии 'c')
- Console input thread (write)
- Voice thread callback (write)

## Паттерны проектирования

### 1. Facade (Фасад)
`HandTrackingClient` - упрощённый интерфейс для сложной системы

### 2. Strategy (Стратегия)
`TrackerManager` - разные алгоритмы трекинга (KCF/CSRT)

### 3. Observer (Наблюдатель)
`VoiceController.on_command_callback` - уведомление о голосовых командах

### 4. Dependency Injection
Все компоненты получают зависимости через конструктор

### 5. Factory (Фабрика)
`TrackerManager._create_tracker()` - создание разных типов трекеров

## Масштабируемость

### Лёгко добавить:

1. **Новый тип трекера**
   - Расширить `TrackerManager._create_tracker()`
   - Добавить в choices в `main.py`

2. **Новый способ ввода команд**
   - Создать новый контроллер (например, `GamepadController`)
   - Добавить callback в `HandTrackingClient`

3. **Новый тип сервера детекции**
   - Создать новый `ServerClient` или расширить существующий
   - Передать в `HybridTracker`

4. **Новые метрики**
   - Расширить `PerformanceMetrics`
   - Обновить `Visualizer` для отображения

## Тестирование

### Unit Tests (рекомендуется создать)

```
tests/
├── test_coordinates.py       # utils.coordinates
├── test_metrics.py            # utils.metrics
├── test_tracker_manager.py   # tracking.tracker_manager
├── test_server_client.py     # network.server_client (mock requests)
└── test_visualizer.py         # ui.visualization
```

### Integration Tests

```
tests/
├── test_hybrid_tracker.py    # tracking + network
└── test_full_pipeline.py     # end-to-end с mock-данными
```

## Производительность

### Узкие места:

1. **LLM inference**: 2-5 секунд (ограничено сервером)
2. **Network latency**: 100-500ms (зависит от сети)
3. **Camera FPS**: обычно 30 FPS
4. **Whisper transcription**: 1-3 секунды

### Оптимизации:

1. **Гибридный трекинг** - классический трекер между LLM вызовами
2. **Асинхронные потоки** - голосовое управление не блокирует видео
3. **Image resizing** - уменьшение размера перед отправкой на сервер
4. **JPEG сжатие** - быстрая передача по сети

## Безопасность потоков

### Thread-safe компоненты:

✅ `PerformanceMetrics` (deque - thread-safe для add)  
✅ `current_prompt` (защищён prompt_lock)  
✅ Все модули используют изоляцию данных  

### Потенциальные проблемы:

⚠️ OpenCV tracker не thread-safe - используется только в main thread  
⚠️ Servo communication - используется только в main thread  

## Расширенные возможности

### Будущие улучшения:

1. **Конфигурация через файл**
   ```python
   config = TrackingConfig.from_yaml("config.yaml")
   ```

2. **Плагины для трекеров**
   ```python
   tracker_registry.register("MOSSE", MosseTracker)
   ```

3. **Web-интерфейс**
   ```python
   from web.dashboard import Dashboard
   dashboard = Dashboard(client)
   dashboard.start()
   ```

4. **Логирование в файл**
   ```python
   from utils.logging_config import setup_file_logging
   setup_file_logging("tracking.log")
   ```
