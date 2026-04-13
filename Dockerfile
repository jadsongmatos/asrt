FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY . .

ENV WHISPER_MODEL=small \
    WHISPER_THREADS=4 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]