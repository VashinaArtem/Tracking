# 🎥 MiniCPM-V-4.5 Real-Time Object Tracking System

Система реал-тайм трекинга объектов с использованием MiniCPM-V-4.5 и веб-камеры.

<div align="center">

**[📋 Project Overview](PROJECT_OVERVIEW.md)** | **[📝 Commands](COMMANDS_CHEATSHEET.md)** | **[⚡ Quick Start](#-quick-start)**

</div>

---

## 🎯 Возможности

✅ Реал-тайм детекция объектов по текстовому описанию  
✅ Работа с веб-камерой  
✅ Динамическая смена промптов во время работы  
✅ Отрисовка bounding boxes на видео  
✅ Две архитектуры: Standalone и Client-Server  
✅ Поддержка CPU и GPU inference  

---

## 🚀 Quick Start

### Выберите версию:

<table>
<tr>
<td width="50%">

### 🖥️ Standalone
**Всё на одной машине**

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r camera_tracker_requirements.txt
python camera_tracker.py
```

✅ Проще  
✅ Автономно  
❌ Требует мощный ПК

[📘 Полная документация](CAMERA_TRACKER_README.md)  
[⚡ Быстрый старт](QUICKSTART.md)

</td>
<td width="50%">

### 🌐 Client-Server
**Сервер + клиент**

```bash
# Linux сервер
python tracking_server.py --host 0.0.0.0

# Windows клиент
python tracking_client.py --server http://IP:8000
```

✅ GPU сервер  
✅ Легкий клиент  
❌ Нужна сеть

[📗 Полная документация](CLIENT_SERVER_README.md)  
[⚡ Быстрый старт](CLIENT_SERVER_QUICKSTART.md)

</td>
</tr>
</table>

---

## 📁 Структура проекта

```
📦 MiniCPM-V Tracking System
│
├── 📖 README.md                          ← Вы здесь
├── 📋 PROJECT_OVERVIEW.md                ← Обзор проекта
├── 📝 COMMANDS_CHEATSHEET.md             ← Шпаргалка команд
│
├── 🖥️  STANDALONE версия
│   ├── camera_tracker.py                 ← Главный скрипт
│   ├── camera_tracker_requirements.txt   ← Зависимости
│   ├── run_camera_tracker.bat            ← Запуск (Windows)
│   ├── CAMERA_TRACKER_README.md          ← Документация
│   └── QUICKSTART.md                     ← Быстрый старт
│
└── 🌐 CLIENT-SERVER версия
    ├── 🖥️  Сервер (Linux)
    │   ├── tracking_server.py            ← Сервер
    │   ├── server_requirements.txt       ← Зависимости
    │   ├── run_server.sh                 ← Запуск
    │   └── test_server.py                ← Тестирование
    │
    ├── 💻 Клиент (Windows)
    │   ├── tracking_client.py            ← Клиент
    │   ├── client_requirements.txt       ← Зависимости
    │   └── run_client.bat                ← Запуск
    │
    └── 📚 Документация
        ├── CLIENT_SERVER_README.md       ← Полная документация
        └── CLIENT_SERVER_QUICKSTART.md   ← Быстрый старт
```

---

## 🎮 Использование

### В консоли вводите промпты:
```
> person              # Найти человека
> red cup             # Найти красную чашку
> laptop computer     # Найти ноутбук
> phone on the table  # Найти телефон на столе
```

### В окне видео используйте клавиши:
- **`q`** - выход
- **`c`** - очистить промпт
- **`s`** - статистика (client-server)

---

## 📊 Сравнение версий

| Характеристика | Standalone | Client-Server |
|---------------|:----------:|:-------------:|
| Простота установки | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Требования к ноутбуку | Высокие | Низкие |
| Скорость (CPU) | 2-5 сек | 2-5 сек |
| Скорость (GPU) | - | 0.5-1.5 сек ⚡ |
| Сеть | Не нужна | Нужна |
| Масштабирование | Нет | Да |

**Рекомендация:** Используйте Standalone для тестирования, Client-Server для продакшена.

---

## 📚 Документация

### 🔰 Начинающим
1. [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - начните отсюда!
2. [QUICKSTART.md](QUICKSTART.md) или [CLIENT_SERVER_QUICKSTART.md](CLIENT_SERVER_QUICKSTART.md)
3. [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md) - шпаргалка команд

### 📖 Подробная документация
- [CAMERA_TRACKER_README.md](CAMERA_TRACKER_README.md) - Standalone версия
- [CLIENT_SERVER_README.md](CLIENT_SERVER_README.md) - Client-Server версия

### 🎯 По задачам

| Я хочу... | Читайте |
|-----------|---------|
| Быстро начать | [QUICKSTART.md](QUICKSTART.md) |
| Понять различия версий | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| Узнать все команды | [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md) |
| Настроить производительность | [CLIENT_SERVER_README.md](CLIENT_SERVER_README.md#-настройка-производительности) |
| Решить проблему | [CAMERA_TRACKER_README.md](CAMERA_TRACKER_README.md#-решение-проблем) |
| Установить на сервер | [CLIENT_SERVER_README.md](CLIENT_SERVER_README.md#-установка) |

---

## 🎯 Примеры использования

### Базовый (Standalone)
```bash
python camera_tracker.py
> person
> laptop computer
```

### Базовый (Client-Server)
```bash
# На Linux сервере
python tracking_server.py --host 0.0.0.0

# На Windows ноутбуке
python tracking_client.py --server http://192.168.1.100:8000
> person
> red cup
```

### Через SSH туннель
```bash
# Windows
ssh -L 8000:localhost:8000 user@server
python tracking_client.py --server http://localhost:8000
```

---

## 💡 Совет: Какую версию выбрать?

### 🖥️ Используйте Standalone если:
- ✅ У вас мощный ноутбук (16GB+ RAM)
- ✅ Не хотите настраивать сервер
- ✅ Нужна максимальная простота
- ✅ Работаете оффлайн

### 🌐 Используйте Client-Server если:
- ✅ Есть доступ к серверу с GPU
- ✅ Ноутбук не очень мощный
- ✅ Нужна высокая скорость (GPU inference)
- ✅ Планируете использовать несколько камер/клиентов

---

## 🔧 Требования

### Общие
- Python 3.9+
- Веб-камера (для клиента)
- OpenCV

### Standalone
- ~16GB RAM (для модели)
- Windows 10/11

### Client-Server
**Сервер (Linux):**
- ~16GB RAM
- Рекомендуется: GPU с CUDA

**Клиент (Windows):**
- ~4GB RAM
- Веб-камера
- Сетевое соединение к серверу

---

## 📈 Производительность

| Конфигурация | Inference Time | Рекомендуемый FPS |
|--------------|----------------|-------------------|
| Standalone CPU | 2-5 сек | 0.5-1 |
| Server CPU | 2-5 сек | 0.5-1 |
| Server GPU | **0.5-1.5 сек** ⚡ | 1-2 |

*Время на кадр может варьироваться в зависимости от сложности сцены и длины промпта*

---

## 🔒 Безопасность

⚠️ **Важно для продакшена:**
- Используйте HTTPS для удаленных подключений
- Добавьте аутентификацию для API
- Настройте firewall правильно
- Не открывайте сервер в интернет без защиты

См. [CLIENT_SERVER_README.md](CLIENT_SERVER_README.md#-безопасность) для деталей.

---

## 🐛 Troubleshooting

### Камера не работает
```bash
# Попробуйте другой ID камеры
python tracking_client.py --camera 1
```

### Не могу подключиться к серверу
```bash
# Проверьте сервер
curl http://SERVER_IP:8000/health

# Проверьте firewall
sudo ufw allow 8000
```

### Медленная работа
- Используйте GPU на сервере (Client-Server)
- Уменьшите FPS: `--fps 0.5`
- Уменьшите качество: `--quality 70`

**Больше решений:** [CAMERA_TRACKER_README.md](CAMERA_TRACKER_README.md#-решение-проблем)

---

## 🌟 Особенности

### MiniCPM-V-4.5
- Мультимодальная модель (vision + language)
- Поддержка grounding (детекция по тексту)
- Возвращает bbox в формате 0-1000 (нормализованные координаты)

### Архитектура
- Многопоточность для неблокирующего ввода
- Thread-safe операции
- Умный frame skipping для оптимальной производительности
- JPEG сжатие для минимизации сетевого трафика (Client-Server)

---

## 📝 Лицензия

Этот проект использует модель MiniCPM-V-4.5. Убедитесь, что соблюдаете лицензию модели.

- [MiniCPM-V-4.5 на HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4_5)

---

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте соответствующий README файл
2. Посмотрите [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)
3. Проверьте логи в консоли
4. Убедитесь что все зависимости установлены

---

## 🎓 Дополнительные ресурсы

- [FastAPI документация](https://fastapi.tiangolo.com/)
- [OpenCV документация](https://docs.opencv.org/)
- [PyTorch документация](https://pytorch.org/docs/)
- [Transformers документация](https://huggingface.co/docs/transformers/)

---

<div align="center">

**Начните прямо сейчас!** 🚀

[Standalone Quick Start](QUICKSTART.md) • [Client-Server Quick Start](CLIENT_SERVER_QUICKSTART.md)

---

*Создано для реал-тайм трекинга объектов с MiniCPM-V-4.5* 🎥🤖

</div>
