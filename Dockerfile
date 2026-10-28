# Everything the agent needs is baked in at build time, so the edge device never needs internet:
# the two List Manuals (parsed), the embedding index and the quantized LLM.
FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --only-binary=llama-cpp-python -r requirements.txt
COPY diag diag
COPY scripts scripts
ARG MODEL=qwen2.5-3b-instruct-q4_k_m.gguf
RUN python scripts/fetch_data.py \
 && python -c "from pathlib import Path; from diag.retrieve import Retriever; Retriever(Path('data/kb'), Path('data/index'))" \
 && find data/models -name '*.gguf' ! -name "$MODEL" -delete && rm -rf data/raw data/models/.cache

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
RUN useradd -r -u 1001 diag
# --chown here, not a later `chown -R`: that would copy the model into a second layer.
COPY --from=build --chown=diag /app /app
ARG MODEL=qwen2.5-3b-instruct-q4_k_m.gguf
ENV DIAG_MODEL=data/models/${MODEL} DIAG_THREADS=4 HF_HUB_OFFLINE=1
USER diag
EXPOSE 8080
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health')"
CMD ["uvicorn", "diag.server:app", "--host", "0.0.0.0", "--port", "8080"]
