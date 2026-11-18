# AI Clip Studio

Автоматический поиск, анализ, отбор и монтаж интересных игровых клипов Twitch
в вертикальный формат 1080×1920 для TikTok и YouTube Shorts.

Проект не скачивает «первые популярные клипы»: каждый клип проходит анализ
аудио и видеоряда, после чего локальная ИИ-модель решает, стоит ли делать из
него ролик. Основной приоритет — качество отбора, а не количество файлов.

Итоговый отчёт по критериям MVP: [docs/FINAL_REPORT.md](<docs/FINAL_REPORT.md>).
Построчная сверка с ТЗ и список отличий: [docs/TZ_COMPLIANCE.md](<docs/TZ_COMPLIANCE.md>).

## Что уже есть

- Twitch-провайдеры: временный публичный **GraphQL** (по умолчанию, без
  собственных credentials) и официальный **Helix** (опционально).
- SQLite как единый источник состояния: статусы, история, попытки, блокировки,
  восстановление после прерывания, миграции с резервной копией.
- Скачивание строго по Twitch Clip ID и URL, с проверкой файла через FFprobe.
- Whisper (faster-whisper) для транскрипции; транскрипт кэшируется в
  `processed/<clip_id>/transcript.json`.
- Локальный анализ кадров через Ollama (`llava`): сетка кадров по всему клипу,
  сцен-чейнджи, dHash-дедупликация без OpenCV, кэш `vision.json`.
- Куратор (`llama3.2`) со строгой JSON-схемой, few-shot примером и повторной
  попыткой; низкая `confidence` переводит кандидата в `REVIEW`.
- Монтаж 1080×1920: игровое видео на размытом фоне, исходный звук.
- Субтитры из пословных таймкодов Whisper.
- Очередь с командами `resume`, `retry`, `status`, `review`, `health`.

Выбор моделей и замеры: [docs/MODELS.md](<docs/MODELS.md>).

## Требования

- Windows 10/11.
- Python 3.10–3.14 (рекомендуется 3.12; 3.14 проверена на этой машине).
- [FFmpeg](https://ffmpeg.org/) — `ffmpeg` и `ffprobe` в `PATH`.
- [Ollama](https://ollama.com) — для локального анализа.
- Модели Ollama:

```bat
ollama pull llama3.2:latest
ollama pull llava:latest
```

`moondream` и `clip-judge` **не используются**: тест на реальных клипах показал
непригодное качество описаний/JSON. Если `llama3.2` не хватит для куратора,
можно попробовать `qwen2.5:7b-instruct` и указать его в `OLLAMA_TEXT_MODEL`.

## Установка

1. Скопируйте `.env.example` в `.env`.
2. Запустите `start.bat`. Он создаст `.venv`, установит зависимости и запустит
   интерактивный мастер.
3. Для Twitch GraphQL ничего заполнять не нужно. Свои `TWITCH_CLIENT_ID` и
   `TWITCH_CLIENT_SECRET` требуются только для `TWITCH_PROVIDER=helix`.

Секреты хранятся только в `.env`; файл исключён из Git.

## Запуск

```bat
start.bat
```

Мастер спросит игру, количество клипов и период. Или напрямую:

```bat
.venv\Scripts\python.exe -m src.main health
.venv\Scripts\python.exe -m src.main run --game "Counter-Strike 2" --count 5 --period week
.venv\Scripts\python.exe -m src.main resume
.venv\Scripts\python.exe -m src.main status
.venv\Scripts\python.exe -m src.main status --clip 2021748262
.venv\Scripts\python.exe -m src.main review
.venv\Scripts\python.exe -m src.main retry --clip 2021748262
.venv\Scripts\python.exe -m src.main cleanup --dry-run
.venv\Scripts\python.exe -m src.main health
```

- `run` ищет новые клипы и продолжает незавершённую очередь по этой игре.
- `resume` продолжает очередь без обращения к Twitch.
- `retry --clip ID` вручную возвращает клип в работу; `--from-stage
  download|analysis|edit` выбирает, с какого этапа начать.
- `review` показывает клипы, которым нужна ручная проверка.
- `cleanup` удаляет старые `frames/`, `audio.wav` и временные файлы
  завершённых клипов; исходники, экспорт и JSON-анализ не трогает.
  `--orphans` разрешает удалить каталоги `processed/`, которых нет в БД.

## Настройка анализа

Основные параметры в `.env`:

- `FRAME_INTERVAL_SEC`, `MIN_FRAMES`, `MAX_FRAMES` — сетка кадров; по умолчанию
  4–10 кадров на клип, интервал растягивается так, чтобы покрыть весь клип.
- `SCENE_THRESHOLD` — порог сцен-чейнджей (0.3).
- `VISION_FRAME_WIDTH` — ширина кадра для vision-модели (768).
- `MIN_SCORE_REVIEW` / `MIN_SCORE_CANDIDATE` — пороги 40/60.
- `MIN_CONFIDENCE` — ниже этого значения кандидат уходит в `REVIEW` (0.35).
- `CURATOR_TEMPERATURE` — температура куратора (0.2).
- `OLLAMA_KEEP_ALIVE` — сколько держать модель в памяти; для 8 GB VRAM
  разумно `2m`.

Кэш `transcript.json`/`vision.json` привязан к fingerprint исходного файла и
модели; при повторном запуске Whisper и vision не пересчитываются.

## Статусы

Обязательные: `DISCOVERED`, `DOWNLOADING`, `DOWNLOADED`, `ANALYZING`,
`ANALYZED`, `REJECTED`, `APPROVED`, `EDITING`, `READY`, `FAILED`.

Дополнительно:

- `REVIEW` — модель не уверена, нужно решение человека или второй проход.
- `ANALYSIS_ERROR` — сбой модели/JSON, а не «плохой клип».

`READY` ставится только после проверки, что файл существует и читается.

## Хранение

```text
raw_clips/<clip_id>_source.mp4
processed/<clip_id>/{audio.wav, transcript.json, frames/, vision.json, ...}
exports/<Категория>/<clip_id>_<Категория>.mp4
data/clips.db
data/backups/clips-YYYYmmdd-HHMMSS.db
logs/app.log
```

Исходный клип не удаляется после монтажа. Временные файлы задачи находятся
в отдельных каталогах `processed/<clip_id>` и `raw_clips/.tmp/<clip_id>`.

## Twitch: важное ограничение

Собственное Twitch-приложение пока невозможно зарегистрировать (2FA), поэтому
по умолчанию используется публичный Client-ID веб-сайта Twitch и GraphQL.
Это **не** официальные credentials проекта. Если Twitch изменит схему,
`health` и `run` сообщат об ошибке, а не вернут пустой список.

Для перехода на официальный API:

```env
TWITCH_PROVIDER=helix
TWITCH_CLIENT_ID=...
TWITCH_CLIENT_SECRET=...
```

Pipeline, downloader и база от провайдера не зависят.

## Тесты

```bat
.venv\Scripts\python.exe -m pytest -q
```

Тесты используют mock-ответы Twitch и FFmpeg-фикстуры; реальные ключи и
интернет для большинства тестов не нужны.

## Диагностика

- `health` проверяет Python, venv, FFmpeg/ffprobe, yt-dlp, Ollama и модели,
  Twitch и состояние базы.
- Модель не найдена: `ollama pull <имя>` и проверьте `OLLAMA_TEXT_MODEL` /
  `OLLAMA_VISION_MODEL` в `.env`.
- Нет `ffprobe`: добавьте папку FFmpeg в `PATH` (в проекте может лежать только
  `ffmpeg.exe`, но `ffprobe.exe` обязателен).
- Ошибка модели у клипа — это `ANALYSIS_ERROR`, а не `REJECTED`; после
  исправления настроек выполните `retry --clip ID`.

## Ограничения MVP

- Публикация в TikTok/YouTube, Telegram-бот, мониторинг стримеров и анализ
  длинных VOD в эту версию не входят.
- Качество отбора ограничено локальными моделями на 8 GB VRAM.
- Авторские права на клипы остаются ответственностью пользователя.
