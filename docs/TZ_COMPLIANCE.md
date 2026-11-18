# Соответствие ТЗ: честная матрица

Дата проверки: 2026-11-18
Ветка: `mvp-hardening`, коммит `35fecd8`.
Полный набор тестов: `207 passed`.

Ниже — не маркетинговое «всё готово», а построчная сверка. Отмечено, что
сделано, что сделано с осознанным отличием и что не делалось.

## Полностью закрыто

| Требование ТЗ | Где |
|---|---|
| Передача `clip_id` и точных путей между этапами, без `max(mtime)` | `src/pipeline.py`, `src/downloader.py`, `src/editor.py` |
| Структурированный результат, исключения, логирование этапов | `pipeline.py`, `logging_setup.py`, `database.status_history` |
| SQLite как источник состояния: статусы, история, попытки, ошибки, экспорт | `src/database.py` (schema v3) |
| Статусы `DISCOVERED…FAILED` + `REVIEW`/`ANALYSIS_ERROR` | `database.py` |
| Миграции с бэкапом, индексы, закрытие соединений | `database.py`, тест `test_migration_from_v2_preserves_data` |
| Повторный запуск не скачивает/не анализирует готовое | downloader skip, `_load_analysis`, `_load_decision` |
| Ручная переобработка | `retry --clip --from-stage` |
| Ошибка одного клипа не останавливает очередь | `tests/test_pipeline_resume.py` |
| `READY` только при существующем проверенном файле | `Database.set_export_path`, `editor._verify_output` |
| Скачивание по ID/URL, проверка файла, retries, `.part`, свободное место | `src/downloader.py` |
| Аудио: Whisper, таймстемпы, кэш, громкость, паузы, смех, крик, шум vs реакция | `src/audio_analyzer.py` |
| Видео: кадры по всему клипу, сцен-чейнджи, dHash без OpenCV, кэш | `src/vision_analyzer.py` |
| Строгий JSON куратора, схема, few-shot, retry, confidence | `src/auto_curator.py` |
| Оценка 20/15/20/15/10/10/10, 0–100, категории, три решения | `src/scoring.py` |
| Ошибка анализа отдельно от отклонения | `mark_analysis_error`, `ANALYSIS_ERROR` |
| Выбор лучшего момента, контекст/реакция, min/max | `src/moment.py` |
| Монтаж 1080×1920, blur/crop, HUD не обрезается в blur, звук, CRF | `src/editor.py` |
| Субтитры ASS PlayRes 1080×1920, 2 строки, accent, RU/EN | `src/subtitler.py`, `test_subtitler_ass.py` |
| Субтитры отключены — рендер работает | `test_editor.py`, `test_pipeline_resume.py` |
| Очистка временных файлов | `src/maintenance.py`, `cleanup` |
| Логирование с `clip_id`, этапом, временем, ошибкой | `src/logging_setup.py` |
| Heartbeat/восстановление зависших задач | `Pipeline._touch`, `Database.recover_stale` |
| Windows/`start.bat`, проверка Python/venv/FFmpeg/yt-dlp/Ollama/моделей | `start.bat`, `src/launcher.py` |
| `.env.example`, секреты только в `.env` | `.env.example`, `.gitignore` |
| GraphQL сохранён, не требует credentials; Helix опционален | `src/twitch/*` |
| Тесты 15 категорий из ТЗ + интеграционный тест | `tests/` (207) |

## Сделано с осознанным отличием

1. **Официальный Twitch Helix.** Реализован и покрыт mock-тестами, но вживую
   не проверен: собственные Client ID/Secret недоступны из-за 2FA. По особому
   условию ТЗ по умолчанию работает GraphQL. Это не скрытая ошибка:
   `health` и `run` явно сообщают о проблемах, а переключение —
   `TWITCH_PROVIDER=helix`.
2. **Двухступенчатый ИИ.** Используются две модели: `llava` для кадров и
   `llama3.2` для нарратива/решения. Отдельного «лёгкого прескрина»,
   который отбирал бы кандидатов до сильной модели, нет: `moondream` в
   замерах давал непригодный текст. Это отличие от «предпочтительной»
   схемы, а не блокер MVP; обоснование в `docs/MODELS.md`.
3. **Хранение.** Добавлены `processed/<clip_id>/analysis.json` и
   `vertical.mp4` (hardlink/копия экспорта). Исходники и экспорт не удаляются.
4. **`MAX_FRAMES`.** 10 кадров по умолчанию вместо формулировки «1–2 секунды
   на кадр»: интервал растягивается на весь клип, чтобы уложиться в 8 GB VRAM.
5. **Внешний API.** Не реализован. Локальный сценарий полностью рабочий;
   внешний режим описан в ТЗ как необязательный. В конфигурации сейчас нет
   `ANALYSIS_MODE=external`, это осознанно не добавлялось без ключа/бюджета.

## Не сделано (по ТЗ — после MVP)

- Этап A: мониторинг стримеров/VOD.
- Этап B: длинные трансляции и нарезка.
- Этап C: Telegram-бот.
- Этап D: автономная фоновая работа.
- Этап E: публикация в TikTok/YouTube.
- Публикация/авторские права — ответственность пользователя.

## Проверено вживую

- Twitch GraphQL: health, поиск, 10 клипов, пагинация.
- yt-dlp: 5 реальных клипов.
- Whisper small CPU: RU/EN, 46 с клип.
- Ollama `llava` 8 кадров и `llama3.2` strict JSON на реальных клипах.
- `tN1R ace` 46 с → `READY`, score 92, момент 20–38 с, 1080×1920, 18.4 с.
- `monesy is backkkk` → `REVIEW`; клипы-реакции → `REJECTED`.
- `start.bat status` и `start.bat run --no-discovery` (resume `APPROVED` → `READY`).
- `analysis.json`/`vertical.mp4` появляются в `processed/<clip_id>/`.

## Что осталось проверить вам

1. Чистое создание `.venv` с нуля на новой машине (в этой сессии проверена
   установка в существующий venv, а не удаление и пересоздание).
2. Живой Helix при появлении credentials.
3. Качество отбора на ваших играх; при необходимости сменить текстовую
   модель на `qwen2.5:7b-instruct`.
4. Push ветки `mvp-hardening` и merge; локальный Git не пушился.
