# Tracking

Реал-тайм трекинг объектов по текстовому/голосовому описанию на базе **MiniCPM-V**, с опциональным управлением pan/tilt-сервоприводами.

Репозиторий объединяет два бывших проекта:

| Было | Стало |
|------|--------|
| `Tracking_server` | [`server/`](server/) |
| `Tracking_client` | [`client/`](client/) |

```
Камера / микрофон  →  Client  →  Server (MiniCPM-V)  →  bbox  →  PID → Servos
                                      ↑
                              GPU (рекомендуется) или CPU/GGUF
```

---

## Структура

```
tracking/
├── README.md
├── server/                              # Inference-сервер
│   ├── tracking_client_server/
│   │   ├── server_gpu_inference/        # ★ основной GPU-сервер (рекомендуется)
│   │   ├── server_cpu_inference/        # GGUF / CPU варианты
│   │   ├── tracking_client.py           # Лёгкий клиент (только камера + bbox)
│   │   └── md_stuff/                    # Документация и requirements
│   └── datasets/MeViS/                  # Датасеты / evaluation
│
└── client/                              # Клиенты + железо
    ├── Minicpm_hand_detect_and_track/   # ★ основной клиент (голос + сервы + hybrid tracker)
    ├── Minicpm_hand/                    # Более ранняя монолитная версия
    ├── Andrew_camera/                   # Калибровка / face-tracking прототип
    └── dji_rs4/                         # Эксперименты с DJI RS4 protocol
```

**Что использовать в проде:**  
`server/tracking_client_server/server_gpu_inference` + `client/Minicpm_hand_detect_and_track`.

---

## Быстрый старт

### 1. Сервер (Linux + GPU)

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

Проверка:

```bash
curl http://localhost:8000/health
```

Альтернатива — CPU/GGUF:

```bash
python server_cpu_inference/tracking_server_2_gguf.py --host 0.0.0.0 --port 8000
```

### 2. Клиент (камера + голос + сервы)

```bash
cd client/Minicpm_hand_detect_and_track
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Тест без железа:

```bash
python main.py --mock-servos --camera 0 --image-size 1080 --server http://SERVER_IP:8000
```

С сервоприводами:

```bash
python main.py \
  --port /dev/tty.usbmodemXXXX \
  --pan-id 1 --tilt-id 2 \
  --camera 0 \
  --server http://SERVER_IP:8000
```

Список камер:

```bash
python list_cameras.py
```

---

## Как это работает

1. **Клиент** захватывает кадр с камеры (и опционально команду с микрофона через Whisper).
2. Кадр + текстовый промпт (`hand`, `red cup`, `person` …) уходят на **сервер**.
3. **MiniCPM-V** возвращает bounding box.
4. Клиент рисует bbox, при наличии железа гоняет **PID → pan/tilt сервы** (Feetech STS).
5. Между вызовами LLM используется hybrid classical tracker для более плавного следования.

Подробнее по клиенту: [`client/Minicpm_hand_detect_and_track/ARCHITECTURE.md`](client/Minicpm_hand_detect_and_track/ARCHITECTURE.md)  
Подробнее по серверу: [`server/tracking_client_server/md_stuff/server.md`](server/tracking_client_server/md_stuff/server.md)

---

## Управление

### Голос / текст
- `hand`, `phone`, `cup`, `person` — начать трекинг
- `clear` / `stop` — сбросить цель
- `quit` / `exit` — выход

### Клавиши в окне видео
| Клавиша | Действие |
|---------|----------|
| `q` | выход |
| `c` | очистить промпт |
| `s` | статистика сервера |

---

## Зависимости

| Компонент | Python | Ключевые пакеты |
|-----------|--------|-----------------|
| GPU-сервер | 3.9+ | `torch`, `transformers`, `fastapi`, `uvicorn` |
| Клиент | 3.8+ | `opencv-contrib-python`, `faster-whisper`, `feetech-servo-sdk`, `requests` |

Файлы requirements:

- Сервер: [`server/tracking_client_server/md_stuff/server_requirements.txt`](server/tracking_client_server/md_stuff/server_requirements.txt)
- Лёгкий клиент: [`server/tracking_client_server/md_stuff/client_requirements.txt`](server/tracking_client_server/md_stuff/client_requirements.txt)
- Основной клиент: [`client/Minicpm_hand_detect_and_track/requirements.txt`](client/Minicpm_hand_detect_and_track/requirements.txt)

Модель: [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)

---

## Железо (опционально)

- USB-камера
- 2× Feetech STS (pan / tilt)
- USB–Serial адаптер
- Микрофон (для голоса)

Калибровка серв:

```bash
# из Andrew_camera или встроенного calibration.py
python calibration.py --port COM3 --pan-id 1 --tilt-id 2
```

Без серв всегда можно работать в `--mock-servos`.

---

## Полезные команды

```bash
# нагрузка GPU
nvtop
nvidia-smi

# кто слушает 8000
lsof -i :8000

# остановить сервер
kill -9 $(lsof -t -i :8000)
```

SSH-туннель, если сервер за NAT:

```bash
ssh -L 8000:localhost:8000 user@server
python main.py --server http://localhost:8000 --mock-servos --camera 0
```

---

## Документация по модулям

| Путь | О чём |
|------|--------|
| [`client/Minicpm_hand_detect_and_track/README.md`](client/Minicpm_hand_detect_and_track/README.md) | Основной клиент: установка, CLI, PID |
| [`client/Minicpm_hand_detect_and_track/ARCHITECTURE.md`](client/Minicpm_hand_detect_and_track/ARCHITECTURE.md) | Архитектура клиента |
| [`server/tracking_client_server/md_stuff/server.md`](server/tracking_client_server/md_stuff/server.md) | GPU-сервер, API, потоки данных |
| [`server/tracking_client_server/server_gpu_inference/markdown/README.md`](server/tracking_client_server/server_gpu_inference/markdown/README.md) | Модули GPU-пакета |
| [`client/Minicpm_hand/README.md`](client/Minicpm_hand/README.md) | Legacy-клиент |
| [`client/Andrew_camera/`](client/Andrew_camera/) | Калибровка / прототипы камеры |

---

## Что не вошло в репозиторий

- виртуальные окружения (`venv/`)
- веса моделей (`.gguf`, `.pt`, …) — скачиваются отдельно
- APK / PCAP / декомпил DJI Ronin (`dji_rs4`) — слишком тяжёлые артефакты

---

## Лицензии

Соблюдайте лицензии используемых моделей и библиотек (MiniCPM-V, llama.cpp, Whisper, OpenCV и др.).
