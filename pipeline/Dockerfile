FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY config config
COPY demo demo
COPY run.py .
ENV PYTHONUNBUFFERED=1 PIPELINE_CONFIG=demo/pipeline.docker.toml
CMD ["python", "run.py", "start"]
