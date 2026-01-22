FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY web ./web

ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app:APP --host 0.0.0.0 --port ${PORT}
