# syntax=docker/dockerfile:1
FROM python:3.13-slim

# uv (fast, respects your pyproject.toml + uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System deps: wget for the model download helper
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the Python 3.13 already in this image; do NOT let uv download a newer
# managed CPython. Project requires >=3.13; onnxruntime 1.20.1 ships a cp313 wheel.
ENV UV_PYTHON_PREFERENCE=only-system

# Install Python deps first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python /usr/local/bin/python3.13

# Copy the application code
COPY . .

# Ensure the entrypoint is executable inside the image
RUN chmod +x /app/entrypoint.sh

# Put the project venv on PATH so 'python'/'streamlit' resolve to it
ENV PATH="/app/.venv/bin:$PATH"

# Knowledge base + embeddings live in a persistent volume
ENV DB_PATH=/app/kb/fifa_worldcup.duckdb \
    EMB_PATH=/app/kb/embeddings.npy \
    EMB_META_PATH=/app/kb/embeddings_meta.json

EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]