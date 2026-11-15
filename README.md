# Tracking

Система, которая **смотрит в камеру, понимает что вы хотите отследить, и сама наводит камеру на объект**.

Вы говорите или пишете что-то вроде «рука», «телефон», «красная кружка» — модель находит это на кадре, обводит рамкой, а сервоприводы (если подключены) поворачивают камеру так, чтобы объект оставался в центре.

В основе — мультимодальная модель [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V-4_5): она принимает картинку и текст и отвечает, *где* на кадре нужный объект.

---

## Зачем так устроено

Тяжёлая модель плохо живёт на ноутбуке рядом с камерой. Поэтому проект разделён на две части:

1. **Сервер** — машина с GPU. Принимает кадр + текстовый запрос, гоняет MiniCPM-V, возвращает координаты рамки (bounding box).
2. **Клиент** — ноутбук / мини-ПК у камеры. Снимает видео, шлёт запросы на сервер, рисует рамку, слушает голос, крутит сервы.

Так слабый клиент может работать быстро, а «мозги» остаются на GPU.

```
микрофон / клавиатура          камера
        │                        │
        ▼                        ▼
   «найди руку»  ─────────►  клиент
                              │
                    JPEG + промпт
                              │
                              ▼
                         GPU-сервер
                       (MiniCPM-V)
                              │
                           bbox
                              │
                              ▼
                    рамка на экране
                    + поворот серв (PID)
```

---

## Что лежит в репозитории

Раньше это были два отдельных репо — `Tracking_server` и `Tracking_client`. Сейчас всё вместе:

| Папка | Что это |
|-------|---------|
| [`server/`](server/) | Inference-сервер и скрипты оценки |
| [`client/`](client/) | Клиенты, голос, сервы, прототипы |

### Сервер — `server/`

| Путь | Смысл |
|------|--------|
| [`server/tracking_client_server/server_gpu_inference/`](server/tracking_client_server/server_gpu_inference/) | **Основной сервер.** FastAPI + MiniCPM-V на GPU. То, что нужно для нормальной работы. |
| [`server/tracking_client_server/server_cpu_inference/`](server/tracking_client_server/server_cpu_inference/) | Вариант на CPU через GGUF / llama.cpp — медленнее, но без видеокарты. |
| [`server/tracking_client_server/tracking_client.py`](server/tracking_client_server/tracking_client.py) | Простой клиент «только камера и рамка», без серв и голоса. |
| [`server/datasets/MeViS/`](server/datasets/MeViS/) | Датасет и скрипты для оценки качества детекции. |

### Клиент — `client/`

| Путь | Смысл |
|------|--------|
| [`client/Minicpm_hand_detect_and_track/`](client/Minicpm_hand_detect_and_track/) | **Основной клиент.** Камера, голос (Whisper), hybrid-трекер, PID → pan/tilt сервы. Модульная архитектура. |
| [`client/Minicpm_hand/`](client/Minicpm_hand/) | Более ранняя монолитная версия того же клиента. Оставлена для истории / сравнения. |
| [`client/Andrew_camera/`](client/Andrew_camera/) | Первые эксперименты: face-tracking и калибровка серв (mac / windows). |
| [`client/dji_rs4/`](client/dji_rs4/) | Отдельные эксперименты с протоколом DJI RS4 (не часть основного пайплайна). |

**Если просто хотите запустить систему:**  
GPU-сервер из `server_gpu_inference` + клиент из `Minicpm_hand_detect_and_track`.

---

## Как это работает на практике

1. На сервере поднимается API на порту `8000`.
2. Клиент открывает камеру и (по желанию) микрофон.
3. Вы говорите или печатаете цель: `hand`, `phone`, `person`…
4. Клиент время от времени отправляет кадр на сервер.
5. Модель отвечает координатами объекта.
6. Между тяжёлыми запросами к модели клиент догоняет объект классическим трекером — картинка остаётся живой, а GPU не дёргают на каждый кадр.
7. Если есть сервы, PID плавно доворачивает камеру к центру объекта.

Без железа тоже можно: режим `--mock-servos` — только видео и рамка, сервы не нужны.

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

Проверка, что жив:

```bash
curl http://SERVER_IP:8000/health
```

Без GPU (медленнее):

```bash
python server_cpu_inference/tracking_server_2_gguf.py --host 0.0.0.0 --port 8000
```

### 2. Клиент

```bash
cd client/Minicpm_hand_detect_and_track
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Посмотреть, какая камера какая:

```bash
python list_cameras.py
```

Только софт, без серв:

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

На Windows порт обычно вида `COM3`.

---

## Управление

**Голосом или текстом в консоли**

| Команда | Действие |
|---------|----------|
| `hand`, `phone`, `cup`, `person`… | начать трекинг этого объекта |
| `clear` / `stop` | сбросить цель |
| `quit` / `exit` | выйти |

**Клавиши в окне видео**

| Клавиша | Действие |
|---------|----------|
| `q` | выход |
| `c` | очистить промпт |
| `s` | статистика сервера |

---

## Железо (не обязательно)

Чтобы камера физически следила за объектом:

- USB-камера
- два сервопривода Feetech STS (pan / tilt)
- USB–Serial адаптер
- микрофон — только если нужен голос

Калибровку диапазонов серв можно сделать через утилиты в `Andrew_camera` или `calibration.py` в основном клиенте. Без железа просто запускайте с `--mock-servos`.

---

## Зависимости

| Часть | Что нужно |
|-------|-----------|
| GPU-сервер | Python 3.9+, PyTorch, transformers, FastAPI, CUDA-GPU |
| Клиент | Python 3.8+, OpenCV, requests; для голоса — Whisper; для серв — feetech-servo-sdk |

Файлы requirements:

- [`server/tracking_client_server/md_stuff/server_requirements.txt`](server/tracking_client_server/md_stuff/server_requirements.txt)
- [`client/Minicpm_hand_detect_and_track/requirements.txt`](client/Minicpm_hand_detect_and_track/requirements.txt)

Модель качается с Hugging Face при первом запуске GPU-сервера: [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5).

---

## Полезные мелочи

```bash
# нагрузка GPU
nvtop          # живой монитор
nvidia-smi     # снимок

# кто занял порт 8000
lsof -i :8000

# если сервер за NAT — туннель на свой ноут
ssh -L 8000:localhost:8000 user@server
python main.py --server http://localhost:8000 --mock-servos --camera 0
```

---

## Куда смотреть дальше

| Документ | О чём |
|----------|--------|
| [`client/Minicpm_hand_detect_and_track/README.md`](client/Minicpm_hand_detect_and_track/README.md) | Детали клиента: CLI, PID, troubleshooting |
| [`client/Minicpm_hand_detect_and_track/ARCHITECTURE.md`](client/Minicpm_hand_detect_and_track/ARCHITECTURE.md) | Как устроены потоки и модули клиента |
| [`server/tracking_client_server/md_stuff/server.md`](server/tracking_client_server/md_stuff/server.md) | Сервер, API, схема данных |
| [`server/tracking_client_server/server_gpu_inference/markdown/README.md`](server/tracking_client_server/server_gpu_inference/markdown/README.md) | Разбор модулей GPU-пакета |

---

## Чего в git нет

- виртуальные окружения
- веса моделей (скачиваются отдельно)
- тяжёлые дампы из `dji_rs4` (APK, PCAP, декомпил)

---

## Лицензии

Соблюдайте лицензии MiniCPM-V, Whisper, llama.cpp, OpenCV и остальных используемых компонентов.
