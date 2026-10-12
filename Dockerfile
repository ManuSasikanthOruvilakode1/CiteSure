FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

# The index is built outside the image (scripts/build_index.py) and mounted at runtime.
# The LLM runs in Ollama on the host, reached through host.docker.internal:
#   docker run -p 8001:8001 -e CITESURE_OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v "$PWD/artifacts:/app/artifacts" -v "$PWD/hf-cache:/app/.cache/huggingface" citesure
EXPOSE 8001
CMD ["uv", "run", "--no-sync", "python", "-m", "citesure.serving.app"]
