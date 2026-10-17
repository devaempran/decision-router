# CPU image running the mock backend: the full decisionLM API with no model and no GPU.
# For real decisions run the service natively on a GPU machine (see README.md).
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir flask numpy waitress
COPY config config
COPY src src
COPY static static
COPY run.py .
ENV DECISIONLM_MOCK=1 DECISIONLM_HOST=0.0.0.0 PORT=5000
EXPOSE 5000
HEALTHCHECK --interval=3s --timeout=3s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health')"
CMD ["python", "run.py"]
