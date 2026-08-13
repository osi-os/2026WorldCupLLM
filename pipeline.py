"""
Automated ingestion pipeline for the FIFA World Cup 2026 RAG project (dlt).

Stages:
  1. Extract : read the Kaggle CSVs from source_data/
  2. Load    : dlt loads the raw dataset tables into DuckDB  (schema: fifa)
  3. Build   : construct the RAG documents from the data
  4. Load    : dlt loads the documents into DuckDB           (table: fifa.documents)
  5. Embed   : refresh the vector embeddings from the documents

Run from the PROJECT ROOT:
    uv run python pipeline.py
"""

import os
import csv
import glob

import dlt

from data_and_ingestion.ingest_docs_for_rag import load_documents

DB_PATH = "fifa_worldcup.duckdb"
SOURCE_DIR = "source_data"

# Raw files we do NOT ingest (predictive / ML modelling artifacts)
EXCLUDE = {
    "match_prediction_features_X",
    "match_prediction_features",
    "match_prediction_targets_y",
}


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


@dlt.source(name="fifa")
def fifa_raw_source(source_dir=SOURCE_DIR):
    """One dlt resource (table) per real CSV in the dataset."""
    for path in sorted(glob.glob(os.path.join(source_dir, "*.csv"))):
        table = os.path.splitext(os.path.basename(path))[0]
        if table in EXCLUDE:
            continue
        yield dlt.resource(
            _csv_rows(path),
            name=table,
            write_disposition="replace",
        )


@dlt.resource(name="documents", write_disposition="replace")
def documents_resource():
    """The RAG knowledge base: one row per document."""
    for doc in load_documents():
        yield {
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "content": doc["content"],
        }


def main():
    pipeline = dlt.pipeline(
        pipeline_name="fifa_worldcup",
        destination=dlt.destinations.duckdb(DB_PATH),
        dataset_name="fifa",
    )

    print("[1-2] Loading raw dataset into DuckDB with dlt...")
    info_raw = pipeline.run(fifa_raw_source())
    print(info_raw)

    print("\n[3-4] Building documents and loading them into DuckDB...")
    info_docs = pipeline.run(documents_resource())
    print(info_docs)

    print("\n[5] Refreshing embeddings...")
    from rag import refresh_embeddings
    refresh_embeddings()

    print("\nPipeline complete. Knowledge base is in", DB_PATH)


if __name__ == "__main__":
    main()