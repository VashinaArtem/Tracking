# prepare_v2 — Подготовка данных для обучения v2

Скрипты для формирования датасетов второго обучения MiniCPM-V на MeViS.

## Схема сплита

```
Train:  1,323 видео (mevis_train.json) + 191 видео (FMD train)  →  ~965k пар
Val:    50 видео (valid_u)  →  30,000 случайных пар             →  mevis_val.json
Test:   90 видео (FMD test) →  ~149k пар (инференс через FMD папки)
```

Из 281 уникальных видео FirstMetricsData: первые 90 (seed=42) → Test, остальные 191 → Train.

## Скрипты

### `prepare_fmd_split.py`

Делит 281 уникальных видео FirstMetricsData на test/train:

```bash
python prepare_fmd_split.py
```

Выход (рядом со скриптом):
- `fmd_test_videos.json`  — 90 video_id для финального теста
- `fmd_train_videos.json` — 191 video_id для добавления в обучение

### `build_val_dataset.py`

Генерирует `mevis_val.json` — 30,000 случайных пар из valid_u (seed=42) в формате ShareGPT для LLaMA-Factory:

```bash
python build_val_dataset.py
# или с явным путём:
python build_val_dataset.py --output ../../../LLaMA-Factory/data/mevis_val.json
```

## Зависимые скрипты (в родительской папке)

После запуска `prepare_fmd_split.py` нужно выполнить в `../`:

```bash
# 1. Извлечь 136 недостающих FMD кадров из JPEGImages.tar
python extract_train_frames.py --test-videos prepare_v2/fmd_test_videos.json

# 2. Сгенерировать расширенный train датасет
python build_train_dataset.py \
    --test-videos prepare_v2/fmd_test_videos.json \
    --output ../../LLaMA-Factory/data/mevis_train_v2.json
```

## Выходные файлы LLaMA-Factory

| Файл | Размер | Назначение |
|---|---|---|
| `LLaMA-Factory/data/mevis_train_v2.json` | ~965k пар | обучение |
| `LLaMA-Factory/data/mevis_val.json` | 30,000 пар | eval_loss каждые 5,000 шагов |
