# GPU Tracking Server - Modular Implementation

Модульная реализация GPU-accelerated tracking server для MiniCPM-V-4.5.

## 📁 Структура проекта

```
server_gpu_inference/
├── __init__.py              # Package initialization, экспорт основных классов
├── tracking_server_gpu.py   # Entry point, запуск сервера
├── models.py                # Pydantic модели (DetectionResponse, ServerStats)
├── config.py                # Конфигурация сервера и константы
├── bbox_parser.py           # Парсинг bounding boxes из ответов модели
├── gpu_manager.py           # Управление GPU (проверка, информация, валидация)
├── tracking_engine.py       # Основной движок inference (TrackingServerGPU)
├── api.py                   # FastAPI endpoints и роуты
├── utils.py                 # Вспомогательные функции (image processing, logging)
└── README.md                # Эта документация
```

## 🎯 Разделение ответственности

### `models.py` - Модели данных
- `DetectionResponse`: Ответ API для детекции объектов
- `ServerStats`: Статистика работы сервера
- Все Pydantic модели для валидации и сериализации

### `config.py` - Конфигурация
- `ServerConfig`: Dataclass с параметрами сервера
- Константы и дефолтные значения
- Валидация конфигурации
- Парсинг аргументов командной строки

### `bbox_parser.py` - Парсинг bounding box
- `BBoxParser`: Извлечение координат из текста
- Поддержка разных форматов (`<box>`, plain numbers)
- Валидация координат
- Легко расширяется новыми стратегиями парсинга

### `gpu_manager.py` - Управление GPU
- Проверка доступности CUDA
- Получение информации о GPU
- Валидация количества GPU
- Утилиты для работы с устройствами

### `tracking_engine.py` - Inference движок
- `TrackingServerGPU`: Основной класс для inference
- Загрузка модели (single/multi-GPU)
- Метод `detect()` для детекции объектов
- Управление статистикой
- Полностью независим от FastAPI

### `api.py` - FastAPI приложение
- `TrackingAPI`: Класс для API endpoints
- REST endpoints: `/`, `/health`, `/stats`, `/detect`
- Обработка запросов и ответов
- Dependency injection для tracking_engine

### `utils.py` - Вспомогательные функции
- Обработка изображений (resize, RGB conversion)
- Форматирование промптов
- Логирование
- Общие утилиты

## 🚀 Использование

### Запуск сервера

```bash
# Базовый запуск (1 GPU)
python -m server_gpu_inference.tracking_server_gpu

# С параметрами
python -m server_gpu_inference.tracking_server_gpu \
    --host 0.0.0.0 \
    --port 8000 \
    --num-gpus 2 \
    --image-size 768 \
    --model openbmb/MiniCPM-V-4_5
```

### Программное использование

```python
from server_gpu_inference import ServerConfig, TrackingServerGPU, create_app

# Создать конфигурацию
config = ServerConfig(
    model_name="openbmb/MiniCPM-V-4_5",
    num_gpus=1,
    max_image_size=768
)

# Инициализировать движок
tracking_server = TrackingServerGPU(config)

# Использовать напрямую
from PIL import Image
image = Image.open("test.jpg")
result = tracking_server.detect(image, "red car")
print(result.bbox)

# Или создать FastAPI app
app = create_app(tracking_server)
```

### Использование отдельных компонентов

```python
# Использовать только BBoxParser
from server_gpu_inference.bbox_parser import BBoxParser

parser = BBoxParser()
bbox = parser.extract_bbox("<box>100 200 300 400</box>")
print(bbox)  # [100, 200, 300, 400]

# Использовать только GPU Manager
from server_gpu_inference.gpu_manager import GPUManager

if GPUManager.check_cuda_available():
    GPUManager.print_gpu_info()
    gpu_count = GPUManager.get_gpu_count()
```

## ✅ Преимущества модульной структуры

1. **Разделение ответственности** - каждый модуль отвечает за свою задачу
2. **Тестируемость** - можно тестировать компоненты независимо
3. **Переиспользование** - компоненты можно использовать в других проектах
4. **Масштабируемость** - легко добавить новые парсеры, стратегии
5. **Читаемость** - понятная структура, легко найти нужный код
6. **Поддержка** - изменения в одном модуле не влияют на другие

## 🧪 Тестирование

```python
# Тест BBoxParser
from server_gpu_inference.bbox_parser import BBoxParser

parser = BBoxParser()
assert parser.extract_bbox("<box>10 20 30 40</box>") == [10, 20, 30, 40]
assert parser.validate_bbox([10, 20, 30, 40]) == True
assert parser.validate_bbox([30, 40, 10, 20]) == False  # invalid order

# Тест конфигурации
from server_gpu_inference.config import ServerConfig

config = ServerConfig(num_gpus=2, max_image_size=1024)
config.validate()  # Проверка валидности
```

## 🔧 Расширение функциональности

### Добавить новый парсер bbox

```python
# В bbox_parser.py
@staticmethod
def _extract_from_json(response: str) -> Optional[List[int]]:
    """Extract bbox from JSON format."""
    import json
    try:
        data = json.loads(response)
        return data.get('bbox')
    except:
        return None
```

### Добавить новый endpoint

```python
# В api.py
async def batch_detect(self, images: List[UploadFile], prompts: List[str]):
    """Batch detection endpoint."""
    results = []
    for image, prompt in zip(images, prompts):
        result = await self.detect(image, prompt)
        results.append(result)
    return results
```

## 📊 Зависимости

```bash
pip install torch transformers fastapi uvicorn pillow pydantic
# Опционально для multi-GPU
pip install accelerate
```

## 🎓 Архитектурные паттерны

- **Dependency Injection**: API получает tracking_server как зависимость
- **Strategy Pattern**: BBoxParser поддерживает разные стратегии парсинга
- **Factory Pattern**: `create_app()` создает настроенное приложение
- **Singleton**: Tracking server инициализируется один раз
- **Separation of Concerns**: Каждый модуль - своя ответственность
