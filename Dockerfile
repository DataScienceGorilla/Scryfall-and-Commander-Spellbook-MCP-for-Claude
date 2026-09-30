# Deckbuilding advisor container. Code + deps live in the image; the RAG databases,
# role index and logs are mounted from the host (see docker-compose.yml), so the
# image stays small and data updates don't need a rebuild.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY requirements-advisor.txt .
RUN pip install -r requirements-advisor.txt

# Bake the ONNX all-MiniLM-L6-v2 embedder (~80 MB) into the image so the first
# query after a deploy doesn't stall on a download.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warmup'])"

COPY advisor_app.py mtg_tools.py role_index.py advisor_ui.html advisor_login.html ./
COPY playbooks/ playbooks/

# Runs as a non-root user; mounted data dirs must be readable (and role_index.json /
# advisor_activity.log writable) by uid 1000.
RUN useradd --uid 1000 --create-home advisor && chown -R advisor /app \
    && cp -r /root/.cache /home/advisor/.cache && chown -R advisor /home/advisor/.cache
USER advisor

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5)"
CMD ["uvicorn", "advisor_app:app", "--host", "0.0.0.0", "--port", "8000"]
