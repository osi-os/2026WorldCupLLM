#!/bin/sh
set -e

mkdir -p /app/kb

# 1. Embedding model (anonymous download from Hugging Face) if missing
if [ ! -f /app/models/Xenova/all-MiniLM-L6-v2/model.onnx ]; then
    echo "[entrypoint] Embedding model not found - downloading..."
    python embedder_scripts/download.py
fi

# 2. Knowledge base (DuckDB + embeddings) if missing.
#    Requires source_data/ to be populated (see README prerequisites).
if [ ! -f "$DB_PATH" ]; then
    echo "[entrypoint] Knowledge base not found - running ingestion pipeline..."
    python pipeline.py
fi

# 3. Serve the app
echo "[entrypoint] Starting Streamlit..."
exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true