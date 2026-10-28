FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf MFG_VECTOR_DEVICE=cpu MFG_TRUST_PROXY=1 MFG_PUBLIC_MODE=1 PORT=10000 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY webapp/requirements-server.txt /app/webapp/requirements-server.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/webapp/requirements-server.txt
COPY rag/*.py /app/rag/
COPY rag/mfg_corpus_20261026_expanded_2.db rag/mfg_chunks_20261028_v2.db /app/rag/
COPY webapp /app/webapp
RUN python /app/webapp/prepare_deploy.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --chdir /app/webapp --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --error-logfile - app:app"]
