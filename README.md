# Legenda STT Server

Servidor OpenAI-compatible para transcrição de áudio (STT) + tradução EN→PT.

## Endpoints

| Endpoint | Descrição |
|----------|-----------|
| `POST /v1/audio/transcriptions` | Transcrição padrão OpenAI |
| `POST /v1/audio/translations` | Tradução any→English via Whisper |
| `POST /v1/audio/dual` | Transcrição + tradução em uma chamada |
| `GET /v1/models` | Lista modelos disponíveis |
| `GET /health` | Health check |

## Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|----------|
| `WHISPER_MODEL` | `small` | Modelo GGML: tiny, base, small, medium, large-v3 |
| `WHISPER_THREADS` | 4 | Número de threads CPU |
| `WHISPER_MODELS_DIR` | `~/.whisper` | Diretório dos modelos GGML |
| `TRANSLATE_TO` | - | Idioma alvo (ex: pt, es, fr) |
| `HOST` | `0.0.0.0` | Host |
| `PORT` | `8000` | Porta |
| `LOG_LEVEL` | `INFO` | Log level |

## Rodar

```bash
pip install -r requirements.txt
WHISPER_MODEL=small TRANSLATE_TO=pt PORT=8123 python main.py
```

O modelo GGML é baixado automaticamente na primeira execução (em `~/.whisper/`).

Para Docker, o modelo é baixado durante o build.

## Docker

```bash
docker build -t legenda-server .
docker run -d -p 8123:8000 -e TRANSLATE_TO=pt legenda-server
```

Ou com compose:
```bash
docker compose up -d --build
```

## Testar

```bash
curl http://localhost:8123/health
curl http://localhost:8123/v1/models
```