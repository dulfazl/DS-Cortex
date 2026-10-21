# DS Cortex API. One uvicorn worker on purpose: run threads and their pauses belong to one process.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ARG WITH_CHROMIUM=1
# libgomp1: LightGBM and XGBoost. Chromium: Kaleido renders static PNG copies of the plots.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && if [ "$WITH_CHROMIUM" = "1" ]; then apt-get install -y --no-install-recommends chromium; fi \
    && rm -rf /var/lib/apt/lists/*
ENV BROWSER_PATH=/usr/bin/chromium

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY agents ./agents
COPY tools ./tools
COPY workflows ./workflows
COPY memory ./memory
COPY prompts ./prompts
COPY scripts ./scripts
COPY eval ./eval

RUN useradd --create-home --uid 10001 cortex && mkdir -p /data && chown -R cortex /data
USER cortex
ENV DATA_DIR=/data/datasets LOG_DIR=/data/logs DB_PATH=/data/ds_cortex.sqlite REPORTS_DIR=/data/reports \
    STATIC_PLOTS=${WITH_CHROMIUM}
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
