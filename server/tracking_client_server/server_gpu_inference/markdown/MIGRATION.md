# Migration Guide: Monolithic → Modular Structure

Руководство по миграции с монолитного файла на модульную структуру.

## 🔄 Что изменилось

### Было (монолитная структура)
```
server_gpu_inference/
└── tracking_server_gpu.py  (364 строки)
```

### Стало (модульная структура)
```
server_gpu_inference/
├── __init__.py              # Package exports
├── __main__.py              # Module runner
├── tracking_server_gpu.py   # Entry point (~80 строк)
├── models.py                # Data models
├── config.py                # Configuration
├── bbox_parser.py           # BBox parsing
├── gpu_manager.py           # GPU utilities
├── tracking_engine.py       # Core engine
├── api.py                   # FastAPI app
├── utils.py                 # Helpers
├── examples.py              # Usage examples
├── README.md                # Documentation
└── MIGRATION.md             # This file
```

## 📦 Изменения в импортах

### До миграции

```python
# Старый монолитный файл
from tracking_server_gpu import (
    TrackingServerGPU,
    DetectionResponse,
    app
)
```

### После миграции

```python
# Новая модульная структура

# Импорт из package
from server_gpu_inference import (
    TrackingServerGPU,
    DetectionResponse,
    ServerConfig,
    create_app
)

# Или импорт из конкретных модулей
from server_gpu_inference.tracking_engine import TrackingServerGPU
from server_gpu_inference.models import DetectionResponse
from server_gpu_inference.config import ServerConfig
from server_gpu_inference.api import create_app
```

## 🚀 Изменения в запуске

### Запуск сервера

#### Было:
```bash
python tracking_server_gpu.py --num-gpus 2 --port 8000
```

#### Стало (3 способа):

**Способ 1: Как модуль**
```bash
python -m server_gpu_inference --num-gpus 2 --port 8000
```

**Способ 2: Через entry point**
```bash
python server_gpu_inference/tracking_server_gpu.py --num-gpus 2 --port 8000
```

**Способ 3: Программно**
```python
from server_gpu_inference import ServerConfig, TrackingServerGPU, create_app
import uvicorn

config = ServerConfig(num_gpus=2, port=8000)
server = TrackingServerGPU(config)
app = create_app(server)
uvicorn.run(app, host=config.host, port=config.port)
```

## 🔧 Изменения в коде

### Инициализация сервера

#### Было:
```python
# Глобальная переменная
tracking_server = None

def main():
    global tracking_server
    tracking_server = TrackingServerGPU(
        model_name="openbmb/MiniCPM-V-4_5",
        num_gpus=1,
        max_image_size=768,
    )
```

#### Стало:
```python
from server_gpu_inference import ServerConfig, TrackingServerGPU

config = ServerConfig(
    model_name="openbmb/MiniCPM-V-4_5",
    num_gpus=1,
    max_image_size=768,
)
tracking_server = TrackingServerGPU(config)
```

### Использование BBox Parser

#### Было:
```python
class TrackingServerGPU:
    def _extract_bbox_from_response(self, response: str):
        # Код парсинга внутри класса
        ...
```

#### Стало:
```python
from server_gpu_inference.bbox_parser import BBoxParser

parser = BBoxParser()
bbox = parser.extract_bbox(response)
is_valid = parser.validate_bbox(bbox)
```

### Работа с GPU

#### Было:
```python
# Все в __init__
if not torch.cuda.is_available():
    raise RuntimeError("...")
available_gpus = torch.cuda.device_count()
# ... много кода
```

#### Стало:
```python
from server_gpu_inference.gpu_manager import GPUManager

if not GPUManager.check_cuda_available():
    raise RuntimeError("...")

GPUManager.print_gpu_info()
gpu_count = GPUManager.get_gpu_count()
```

### FastAPI App

#### Было:
```python
# Глобальная переменная
tracking_server = None

app = FastAPI(...)

@app.get("/detect")
async def detect(...):
    result = tracking_server.detect(...)
    return result
```

#### Стало:
```python
from server_gpu_inference import create_app, TrackingServerGPU

tracking_server = TrackingServerGPU(config)
app = create_app(tracking_server)

# Или для кастомизации
from server_gpu_inference.api import TrackingAPI

api = TrackingAPI()
api.set_tracking_server(tracking_server)
app = api.app
```

## 🧪 Тестирование после миграции

### Проверка импортов

```python
# Проверить что все импортируется
from server_gpu_inference import (
    TrackingServerGPU,
    DetectionResponse,
    ServerConfig,
    create_app
)
print("✓ All imports work")
```

### Проверка функциональности

```bash
# 1. Запустить сервер
python -m server_gpu_inference --port 8000

# 2. В другом терминале - проверить endpoints
curl http://localhost:8000/health
curl http://localhost:8000/stats

# 3. Тест детекции
curl -X POST http://localhost:8000/detect \
  -F "image=@test.jpg" \
  -F "prompt=person"
```

### Unit тесты

```python
# test_components.py
from server_gpu_inference.bbox_parser import BBoxParser
from server_gpu_inference.config import ServerConfig

def test_bbox_parser():
    parser = BBoxParser()
    bbox = parser.extract_bbox("<box>10 20 30 40</box>")
    assert bbox == [10, 20, 30, 40]
    assert parser.validate_bbox(bbox) == True

def test_config_validation():
    config = ServerConfig(num_gpus=2)
    config.validate()  # Should not raise
    
    try:
        bad_config = ServerConfig(num_gpus=10)
        bad_config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_bbox_parser()
    test_config_validation()
    print("✓ All tests passed")
```

## 📋 Checklist для миграции существующего кода

- [ ] Обновить импорты с `tracking_server_gpu` на `server_gpu_inference`
- [ ] Заменить прямое создание `TrackingServerGPU()` на использование `ServerConfig`
- [ ] Обновить скрипты запуска на `python -m server_gpu_inference`
- [ ] Обновить зависимости в requirements.txt (если нужно)
- [ ] Запустить тесты для проверки совместимости
- [ ] Обновить документацию проекта
- [ ] Убедиться что все endpoints работают
- [ ] Проверить логирование и статистику

## ⚠️ Возможные проблемы

### Проблема: Не находит модуль
```
ModuleNotFoundError: No module named 'server_gpu_inference'
```

**Решение:** Убедитесь что запускаете из правильной директории:
```bash
cd /path/to/tracking_client_server
python -m server_gpu_inference
```

### Проблема: Импорты не работают
```
ImportError: cannot import name 'TrackingServerGPU'
```

**Решение:** Проверьте `__init__.py`:
```python
from .tracking_engine import TrackingServerGPU
```

### Проблема: Глобальная переменная tracking_server
```
NameError: name 'tracking_server' is not defined
```

**Решение:** Используйте dependency injection через `create_app()`:
```python
server = TrackingServerGPU(config)
app = create_app(server)  # Передаем server как зависимость
```

## 🎯 Преимущества новой структуры

1. **Модульность**: Каждый компонент независим
2. **Тестируемость**: Легко писать unit-тесты для каждого модуля
3. **Расширяемость**: Просто добавлять новую функциональность
4. **Читаемость**: Легче ориентироваться в коде
5. **Переиспользование**: Компоненты можно использовать отдельно
6. **Поддержка**: Изменения локализованы в одном модуле

## 📚 Дополнительные ресурсы

- `README.md` - Полная документация модулей
- `examples.py` - Примеры использования
- `tracking_server_gpu.py` - Entry point с примерами аргументов
