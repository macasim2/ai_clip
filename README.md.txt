markdown
# AI Clip Studio

Автоматический отбор и монтаж интересных игровых клипов Twitch
в вертикальный формат 1080x1920.

## Быстрый старт (Windows)

1. Установите Python 3.10+, FFmpeg, yt-dlp. Добавьте их в `PATH`.
2. Установите [Ollama](https://ollama.com), затем:
ollama pull qwen2.5:7b
ollama pull moondream

text
3. Скопируйте `.env.example` → `.env`.
4. Запустите `start.bat`.

## CLI
python -m src.main health
python -m src.main run --game "Counter-Strike 2" --count 5 --period week
python -m src.main status
python -m src.main retry

text

## Провайдеры Twitch

- `graphql` — по умолчанию, публичный Client-ID веб-клиента, 2FA не требуется.
- `helix` — официальный, требует `TWITCH_CLIENT_ID` и `TWITCH_CLIENT_SECRET`.

Переключение — через `TWITCH_PROVIDER` в `.env`.

Если `graphql` вернёт `GraphQL errors`, значит Twitch изменил схему:
обновите `GAME_CLIPS` в `src/twitch/graphql_provider.py` по актуальному
запросу из DevTools.

## Тесты
pytest -q tests/

text

## Структура хранения
raw_clips/<clip_id><title>.mp4
processed/<clip_id>/{audio.wav, transcript.json, frames/, vision.json, subtitles.srt}
exports/<Категория>/<clip_id><Категория>.mp4
data/clips.db
logs/app.log