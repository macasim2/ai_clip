# Финальный отчёт MVP

Дата: 2026-11-18
Ветка: `mvp-hardening`
Коммиты: `da15a02` (baseline) → `9d1c107` → `b3d37a5` → `28199f4` → финальный коммит этапа 6 (см. `git log`).
Среда: Windows, Python 3.14.6 в `.venv`, FFmpeg/ffprobe в PATH, Ollama 0.40.13,
`llama3.2:latest` + `llava:latest`.

## Критерии готовности MVP

| # | Критерий | Статус | Доказательство |
|---|---|---|---|
| 1 | Запуск через `start.bat` | ✅ | `cmd /c start.bat status` и `start.bat run ... --no-discovery` завершились с кодом 0; зависимости ставятся один раз по маркеру `.venv\.deps_ok` |
| 2 | Выбор игры и количества роликов | ✅ | интерактивный мастер `launcher.py`; CLI `run --game ... --count ...` |
| 3 | Twitch через официальный интерфейс | ⚠️ | по особому условию по умолчанию используется публичный **GraphQL** (без своих credentials); официальный Helix реализован и покрыт mock-тестами, но не проверен вживую без Client ID/Secret |
| 4 | Обработанные клипы не скачиваются повторно | ✅ | downloader проверяет `local_path`/канонический файл; тесты `test_downloader.py` |
| 5 | Каждый клип обрабатывается независимо | ✅ | `tests/test_pipeline_resume.py`: сбой одного клипа → `FAILED`, следующий → `READY` |
| 6 | ИИ анализирует несколько кадров и аудио | ✅ | реальный клип `tN1R ace`: 8 кадров, 131 слово, аудиособытия; кэши `vision.json`/`transcript.json` |
| 7 | Оценка по содержанию, а не просмотрам | ✅ | views не входят в промпт/веса; кандидат `tN1R ace` score 92, клип-реакция `monesy` score 44 → `REVIEW` |
| 8 | ИИ определяет лучший момент | ✅ | `moment.py`; для `tN1R ace` выбран 20–38 с из 46 с |
| 9 | Монтаж использует этот момент | ✅ | экспорт 18.4 с из 46 с, `best_start/best_end` в БД, `moment` в результате |
| 10 | Формат 1080×1920 | ✅ | `ffprobe`: 1080×1920, аудио на месте; `tests/test_editor.py` |
| 11 | Субтитры синхронизированы | ✅ | ASS из пословных таймкодов Whisper; визуальная проверка кадра |
| 12 | Ошибка клипа не останавливает очередь | ✅ | тесты очереди, реальные прогоны 3+2 клипов |
| 13 | Повторный запуск продолжает состояние | ✅ | `resume` из `ANALYZED`/`APPROVED`/`EDITING` без повторного ИИ; `start.bat run --no-discovery` доделал `APPROVED`-клип |
| 14 | В БД сохраняется история | ✅ | `status --clip` показывает `DISCOVERED → … → READY` с этапами и ошибками |
| 15 | Результат проверяется без правки кода | ✅ | `status`, `review`, `cleanup`, `health`, файлы в `exports/<Категория>/` |

## Тесты

```
.venv\Scripts\python.exe -m pytest -q
207 passed in 10.81s
```

Покрыто: конфигурация, миграции SQLite v3 с бэкапом, очередь/claim/lock,
Twitch GraphQL/Helix на mock-ответах, downloader, аудио-события и кэш,
vision-сетка/сцены/dHash/кэш, куратор со строгой схемой, scoring, выбор
момента, ASS/SRT-субтитры, редактор с реальным FFmpeg, очистка, полный
конвейер с реальным рендером и resume.

## Реальные прогоны

| Клип | Результат |
|---|---|
| `tN1R ace`, 46 с | `READY`, score 92, момент 20–38 с, экспорт 18.4 с, 1080×1920, аудио |
| `monesy is backkkk`, 15.6 с | `REVIEW`, score 44 (нет ясного события) |
| `727687811` «что», 9.85 с | `REJECTED`, score 48 (реакция без события) |
| `3856112709` «огонек продли», 6 с | `REJECTED`, score 36 |
| `2021748262` «сстрогий про себя», 5 с | при старой калибровке score 68; после нового гейта такой случай уходит в `REVIEW`, а не в тихое отклонение |
| `stage6test` (resume) | `APPROVED` → `READY` через `start.bat run --no-discovery`, 12.9 с, 1080×1920 |

## Что проверено вживую

- Живой Twitch GraphQL: health, поиск игры, 10 клипов, пагинация.
- yt-dlp: скачивание 5 реальных клипов.
- Whisper `small` CPU/int8: расшифровка RU/EN, 46 с клип за ~9 с.
- Ollama `llava`: 8 кадров; `llama3.2` со strict JSON; реальные решения.
- FFmpeg: blur-режим, HUD не обрезан, 1080×1920, аудио, ASS-субтитры.
- `start.bat`: установка зависимостей, маркер, `status`, `run --no-discovery`.
- Повторный запуск: кэш и resume.

## Ограничения

- Публичный GraphQL может измениться; это осознанное временное ограничение.
  Официальный Helix готов, но требует собственных credentials (2FA).
- Качество отбора ограничено `llama3.2` 3B. При необходимости:
  `ollama pull qwen2.5:7b-instruct` и `OLLAMA_TEXT_MODEL=qwen2.5:7b-instruct`.
- 8 GB VRAM: по умолчанию `MAX_PARALLEL=1`, 4–10 кадров на клип.
- Длинные 600-секундные VOD не входят в MVP; архитектура очереди готова к
  будущим этапам A–E, но они не реализованы.
- Публикация, Telegram-бот и мониторинг стримеров не реализованы (по ТЗ —
  после стабильного MVP).
- Авторские права на клипы остаются ответственностью пользователя.

## Как запустить

```bat
start.bat
```

или:

```bat
.venv\Scripts\python.exe -m src.main health
.venv\Scripts\python.exe -m src.main run --game "Counter-Strike 2" --count 5 --period week
.venv\Scripts\python.exe -m src.main status
.venv\Scripts\python.exe -m src.main review
.venv\Scripts\python.exe -m src.main resume
.venv\Scripts\python.exe -m src.main cleanup --dry-run
```
