# Demo site: search, written answers and private PDF uploads. Deliberately does NOT
# `pip install .` (that would pull torch via sentence-transformers); it needs only
# the light c2k modules.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-web.txt

# Bake the models the site needs into the image, so the running site never
# downloads anything: the small embedding model (matches a question's meaning)
# and the tokenizer used to split uploaded documents. Each download has a time
# limit and retries, because a stalled download would otherwise hang the build.
ENV C2K_EMBED_CACHE=/app/.cache/fastembed TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN mkdir -p /app/.cache/tiktoken && for i in 1 2 3 4 5 6; do \
      timeout 300 python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')" && break; \
      echo "embedding model download failed (try $i), retrying"; sleep 15; \
    done && test -n "$(ls /app/.cache/fastembed)"
RUN for i in 1 2 3 4 5 6; do \
      timeout 120 python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && break; \
      echo "tokenizer download failed (try $i), retrying"; sleep 10; \
    done && test -n "$(ls /app/.cache/tiktoken)" && chown -R user:user /app/.cache

COPY c2k ./c2k
COPY config ./config
COPY prompts ./prompts
COPY .streamlit ./.streamlit
COPY web ./web

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER user
EXPOSE 7860
CMD ["streamlit", "run", "web/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxUploadSize=25"]
