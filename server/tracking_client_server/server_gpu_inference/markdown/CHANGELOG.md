# Changelog

## [1.0.1] - 2026-02-03

### Changed
- **DEFAULT_MODEL**: Изменена дефолтная модель с `openbmb/MiniCPM-V-2_6` на `openbmb/MiniCPM-V-4_5`
  - Обновлен `config.py`: константа `DEFAULT_MODEL`
  - Обновлен `tracking_server_gpu.py`: аргумент `--model` теперь использует `DEFAULT_MODEL`
  - Обновлен `examples.py`: все примеры используют новую модель
  - Обновлена документация: `README.md`, `MIGRATION.md`, `SUMMARY.md`

### Impact
- При запуске без параметра `--model` будет использоваться MiniCPM-V-4.5 вместо 2.6
- Все примеры кода в документации обновлены
- Обратная совместимость сохранена: можно явно указать старую модель через `--model openbmb/MiniCPM-V-2_6`

---

## [1.0.0] - 2026-02-03

### Added
- Модульная архитектура tracking server
- Разделение на компоненты:
  - `models.py` - Data models
  - `config.py` - Configuration management
  - `bbox_parser.py` - BBox extraction
  - `gpu_manager.py` - GPU utilities
  - `tracking_engine.py` - Core inference engine
  - `api.py` - FastAPI endpoints
  - `utils.py` - Helper functions
- Комплексная документация
- Unit тесты
- Примеры использования

### Changed
- Переход от монолитной структуры (1 файл) к модульной (9+ модулей)
- Улучшенная тестируемость и расширяемость
