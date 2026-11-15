# Architecture Documentation

Архитектура модульного GPU Tracking Server.

## 🏗️ Общая архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Point                          │
│              tracking_server_gpu.py                     │
│            (CLI args → Config → Init)                   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌─────────────┐
│   Config     │          │    API      │
│   Module     │          │   Module    │
└──────┬───────┘          └──────┬──────┘
       │                         │
       │  ┌──────────────────────┘
       │  │
       ▼  ▼
┌──────────────────────────────────────┐
│      Tracking Engine                 │
│    (Core Business Logic)             │
└───────┬──────────────────────────────┘
        │
        ├─────────┬─────────┬─────────┐
        ▼         ▼         ▼         ▼
   ┌────────┐ ┌──────┐ ┌────────┐ ┌─────┐
   │  GPU   │ │ BBox │ │ Utils  │ │Model│
   │Manager │ │Parser│ │        │ │     │
   └────────┘ └──────┘ └────────┘ └─────┘
```

## 📦 Модули и их ответственность

### 1. **tracking_server_gpu.py** - Entry Point
**Ответственность:** Точка входа в приложение
- Парсинг аргументов командной строки
- Создание конфигурации
- Инициализация компонентов
- Запуск сервера

**Зависимости:**
- `config` → создание конфигурации
- `tracking_engine` → создание сервера
- `api` → создание FastAPI app

**Вызывается:** Из командной строки или как модуль

---

### 2. **config.py** - Configuration
**Ответственность:** Управление конфигурацией
- Определение параметров сервера
- Валидация конфигурации
- Константы и defaults
- Создание config из CLI args

**Зависимости:** Нет (базовый модуль)

**Используется:**
- `tracking_server_gpu` → создание config
- `tracking_engine` → использование параметров

---

### 3. **models.py** - Data Models
**Ответственность:** Модели данных
- `DetectionResponse` - ответ API
- `ServerStats` - статистика сервера
- Pydantic валидация
- Сериализация/десериализация

**Зависимости:** `pydantic`

**Используется:**
- `api` → response models
- `tracking_engine` → создание responses

---

### 4. **tracking_engine.py** - Core Engine ⭐
**Ответственность:** Основная бизнес-логика
- Загрузка модели (single/multi-GPU)
- Inference метод `detect()`
- Управление статистикой
- Обработка ошибок

**Зависимости:**
- `config` → параметры
- `models` → DetectionResponse
- `bbox_parser` → извлечение координат
- `gpu_manager` → управление GPU
- `utils` → вспомогательные функции

**Используется:**
- `api` → вызов detect()
- Можно использовать standalone без API

**Ключевые методы:**
```python
__init__(config: ServerConfig)  # Инициализация
detect(image, prompt) → DetectionResponse  # Детекция
get_stats() → dict  # Статистика
```

---

### 5. **api.py** - FastAPI Application
**Ответственность:** REST API endpoints
- Определение endpoints
- Обработка HTTP запросов
- Валидация input/output
- Error handling

**Зависимости:**
- `models` → response models
- `tracking_engine` → вызов detect()
- `utils` → image processing

**Endpoints:**
- `GET /` - информация о сервере
- `GET /health` - health check
- `GET /stats` - статистика
- `POST /detect` - детекция объектов

**Используется:** `tracking_server_gpu` → запуск сервера

---

### 6. **bbox_parser.py** - BBox Extraction
**Ответственность:** Извлечение bounding boxes
- Парсинг из `<box>` тегов
- Fallback стратегии
- Валидация координат

**Зависимости:** `re` (regex)

**Используется:** `tracking_engine` → extract_bbox()

**Strategy Pattern:**
```python
extract_bbox(response)
  ├─> _extract_from_box_tags()     # Strategy 1
  └─> _extract_consecutive_numbers() # Strategy 2 (fallback)
```

---

### 7. **gpu_manager.py** - GPU Utilities
**Ответственность:** Работа с GPU
- Проверка доступности CUDA
- Получение информации о GPU
- Валидация количества GPU
- Утилиты для device/dtype

**Зависимости:** `torch`

**Используется:** `tracking_engine` → GPU management

**Все методы статические** - utility class

---

### 8. **utils.py** - Helper Functions
**Ответственность:** Вспомогательные функции
- Обработка изображений (resize, RGB)
- Форматирование промптов
- Логирование
- Pretty printing

**Зависимости:** `PIL`

**Используется:**
- `tracking_engine` → image processing
- `api` → image conversion

---

## 🔄 Поток данных

### Инициализация сервера
```
CLI Args
  ↓
parse_arguments()
  ↓
create_config_from_args() → ServerConfig
  ↓
TrackingServerGPU(config)
  ├─> GPUManager.check_cuda_available()
  ├─> Load model (transformers)
  └─> Init statistics
  ↓
create_app(tracking_server) → FastAPI app
  ↓
uvicorn.run(app)
```

### Detection Request Flow
```
POST /detect (image, prompt)
  ↓
api.detect()
  ├─> Read image bytes
  ├─> PIL.Image.open()
  └─> ensure_rgb()
  ↓
tracking_engine.detect(image, prompt)
  ├─> resize_image_if_needed()
  ├─> format_grounding_question()
  ├─> model.chat() [GPU inference]
  ├─> BBoxParser.extract_bbox()
  └─> Update statistics
  ↓
DetectionResponse(success, bbox, time, prompt)
  ↓
JSON response to client
```

## 🎯 Design Patterns

### 1. **Dependency Injection**
```python
# API не создает tracking_engine, а получает его
api = TrackingAPI()
api.set_tracking_server(tracking_server)
```

### 2. **Strategy Pattern**
```python
# BBoxParser - разные стратегии парсинга
BBoxParser.extract_bbox()
  - Strategy 1: Box tags
  - Strategy 2: Consecutive numbers
  - Easy to add Strategy 3, 4...
```

### 3. **Factory Pattern**
```python
# create_app() фабрика для FastAPI
app = create_app(tracking_server)
```

### 4. **Configuration Object**
```python
# Все параметры в одном объекте
config = ServerConfig(...)
server = TrackingServerGPU(config)
```

### 5. **Separation of Concerns**
```
Presentation Layer (api.py)
     ↓
Business Logic Layer (tracking_engine.py)
     ↓
Utility Layer (gpu_manager, bbox_parser, utils)
     ↓
Data Layer (models.py)
```

## 🔒 Принципы SOLID

### Single Responsibility Principle ✅
Каждый модуль имеет одну ответственность:
- `api.py` - только HTTP endpoints
- `tracking_engine.py` - только inference
- `bbox_parser.py` - только парсинг bbox

### Open/Closed Principle ✅
Модули открыты для расширения, закрыты для модификации:
- Можно добавить новый парсер в `BBoxParser` не меняя существующие
- Можно добавить новый endpoint не меняя `tracking_engine`

### Liskov Substitution Principle ✅
Можно заменить `TrackingServerGPU` на другую реализацию:
```python
class TrackingServerCPU:
    def detect(self, image, prompt) -> DetectionResponse:
        # CPU implementation
```

### Interface Segregation Principle ✅
Клиенты не зависят от методов, которые не используют:
- API использует только `detect()` и `get_stats()`
- BBoxParser независим от GPU логики

### Dependency Inversion Principle ✅
Зависимости через абстракции:
- API зависит от `DetectionResponse` (абстракция), не от implementation
- Config передается как параметр, не хардкодится

## 📊 Статистика модульности

| Метрика | Значение |
|---------|----------|
| Всего модулей | 9 |
| Строк в самом большом модуле | ~230 (tracking_engine) |
| Средний размер модуля | ~120 строк |
| Циклических зависимостей | 0 |
| Тестируемых компонентов | 6 |

## 🧪 Тестирование

### Unit Tests
Каждый модуль тестируется независимо:
```python
test_bbox_parser()  # Без GPU, без API
test_config()       # Только валидация
test_models()       # Только Pydantic
test_utils()        # Только утилиты
```

### Integration Tests
```python
test_tracking_engine()  # Engine + BBoxParser + GPUManager
test_api_endpoints()    # API + Engine
```

### End-to-End Tests
```bash
# Запустить сервер и тест через HTTP
python -m server_gpu_inference &
curl -X POST http://localhost:8000/detect ...
```

## 🚀 Расширение

### Добавить новый парсер
```python
# bbox_parser.py
@staticmethod
def _extract_from_json(response: str):
    # Новая стратегия
    pass
```

### Добавить кэширование
```python
# caching.py (новый модуль)
class DetectionCache:
    def get(self, image_hash, prompt):
        pass
    
    def set(self, image_hash, prompt, result):
        pass

# tracking_engine.py
def detect(self, image, prompt):
    cache_key = hash_image(image)
    if cached := self.cache.get(cache_key, prompt):
        return cached
    # ... inference
```

### Добавить метрики
```python
# metrics.py (новый модуль)
class MetricsCollector:
    def record_inference_time(self, time):
        pass
    
    def record_success(self):
        pass
    
    def get_metrics(self):
        pass
```

## 📚 Дополнительно

- `README.md` - Документация по использованию
- `MIGRATION.md` - Гайд по миграции
- `examples.py` - Примеры использования
- `tests.py` - Unit тесты
