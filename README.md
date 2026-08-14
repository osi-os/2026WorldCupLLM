# FIFA World Cup 2026 RAG Assistant

A question-answering assistant that lets you ask natural-language questions about
the 2026 FIFA World Cup (match results, player stats, team records, venues, and
tournament-wide leaders) and get grounded answers pulled straight from a
tournament dataset. It's built as a Retrieval-Augmented Generation (RAG) system:
it retrieves the most relevant documents from a knowledge base and hands them to
an LLM (Anthropic's Claude) to write the answer.

---

## Problem description

When a big tournament wraps up, the stats exist but they're scattered across match
pages, player tables, group standings, and venue lists. If you want to answer
something specific like *"how did Mexico do against South Africa?"* or *"which team
scored the most goals in the group stage?"*, you end up clicking through a bunch of
pages and piecing it together yourself. A plain LLM isn't much help either: the
tournament happened after the model's training cutoff, so if you just ask it
directly, it either doesn't know or makes something up.

This project solves that by building a RAG system over a structured dataset of the
2026 World Cup. Every question is answered **only** from the tournament data, so
you get specific, grounded answers (with the exact scores, dates, and names)
instead of a model guessing from stale training knowledge. The result is a single
chat interface where you can ask anything about the tournament and trust that the
answer comes from the data.

---

## Dataset

The data comes from the [FIFA World Cup 2026 dataset on Kaggle](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset)
(`mominullptr/fifa-world-cup-2026-dataset`). It's a set of CSVs covering all 104
matches, ~1,200 players, 48 teams, 16 venues, match events, lineups, and team
stats. I pull it down automatically with `kagglehub` (see setup below), so the
dataset is fully accessible and reproducible and you don't have to manually
download anything.

I intentionally **exclude** the dataset's prediction/ML files
(`match_prediction_*`) from the knowledge base, since I wanted the assistant to
report what actually happened, not modeled predictions.

---

## How it works (architecture)

The flow, end to end:

1. **Ingest.** `pipeline.py` (built with **dlt**) loads the raw CSVs into a DuckDB
   database, then builds ~1,400 text "documents" (one per match, player, team, and
   venue, plus pre-computed summary documents) and loads those into DuckDB as the
   knowledge base.
2. **Index.** `rag.py` builds two search indexes over the documents: a keyword
   index (minsearch) and a vector index (ONNX `all-MiniLM-L6-v2` embeddings).
3. **Retrieve.** For each question, it runs **hybrid search** (keyword + vector,
   fused with Reciprocal Rank Fusion) and takes the top 5 documents.
4. **Generate.** Those documents become the context in a prompt to Claude
   (`claude-haiku-4-5`), which writes the answer using only that context.
5. **Serve.** A Streamlit app is the chat interface.
6. **Monitor.** Every question/answer is logged to Postgres, users can leave
   thumbs up/down, and Grafana visualizes it all.

---

## Project structure

```
2026WorldCupLLM/
├── app.py                          # Streamlit chat interface
├── rag.py                          # retrieval + RAG logic, index building
├── db.py                           # Postgres logging (conversations, feedback)
├── pipeline.py                     # dlt ingestion pipeline (CSVs -> DuckDB -> KB)
├── data_and_ingestion/
│   ├── import_data.py              # downloads the Kaggle dataset
│   ├── ingest_docs_for_rag.py      # builds documents from the CSVs
│   └── summary_docs.py             # builds pre-computed summary documents
├── embedder_scripts/
│   ├── download.py                 # downloads the ONNX embedding model
│   └── embedder.py                 # ONNX all-MiniLM-L6-v2 embedder
├── evaluation/
│   ├── generate_ground_truth.py    # LLM-generated eval questions
│   ├── evaluate_retrieval.py       # text vs vector vs hybrid
│   └── llm_judge.py                # prompt comparison via LLM-as-judge
├── grafana/                        # datasource + dashboard provisioning
├── docker-compose.yml              # app + postgres + grafana
├── Dockerfile                      # app container
├── entrypoint.sh                   # container startup (model + KB + serve)
├── pyproject.toml / uv.lock        # pinned dependencies
└── .env                            # API keys + DB config (not committed)
```

---

## Setup and how to run

You'll need an [Anthropic API key](https://console.anthropic.com/) and a
[Kaggle account](https://www.kaggle.com/) (for the dataset download). Everything
else is handled by the project.

### 1. Environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=worldcup
POSTGRES_USER=user
POSTGRES_PASSWORD=password
```

### 2. Get the dataset

This is the one step the container can't do on its own (it needs Kaggle auth), so
run it on the host first:

```bash
uv sync                                             # install dependencies (Python 3.13)
uv run python data_and_ingestion/import_data.py     # downloads CSVs into source_data/
```

### Option A: Run everything with Docker (recommended)

This brings up the app, Postgres, and Grafana together. On first launch the
container downloads the embedding model and builds the knowledge base
automatically (this takes a minute or two), then starts serving.

```bash
docker compose up --build
```

- App: http://localhost:8501
- Grafana: http://localhost:3000 (login `admin` / `admin`)

### Option B: Run locally

```bash
uv run python embedder_scripts/download.py   # download the ONNX model
uv run python pipeline.py                     # build the knowledge base + embeddings
docker compose up -d postgres grafana         # start the monitoring services
uv run streamlit run app.py                   # start the app at localhost:8501
```

### Keeping the data fresh

If the dataset gets updated, re-running the pipeline refreshes everything: the raw
tables, the documents, and the embeddings. The embedding cache is content-aware,
so it recomputes automatically when the data changes.

```bash
uv run python data_and_ingestion/import_data.py && uv run python pipeline.py
```

All dependency versions are pinned in `uv.lock`, so the environment is fully
reproducible.

---

## Retrieval evaluation

I created an `evaluation/` folder to compare different retrieval methods (text,
vector, and hybrid) against 900 ground-truth questions generated from the
documents.

```
method        hit_rate         mrr
----------------------------------
text             0.904       0.876
vector           0.949       0.917
hybrid           0.962       0.912
```

The script's automatic pick was vector (best MRR), but after looking at the table
I decided to go with the **hybrid** approach as my default search type. My
reasoning: the RAG pipeline passes the top 5 documents to Claude as context, and
hybrid has the highest **hit rate**. The higher hit rate means the correct
document lands in Claude's context more often, which is what actually drives answer
quality. MRR matters most when only the top 1-2 results are used, where rank
position is critical. Since Claude reads all 5 documents that get passed through,
hit rate is the metric that matters more here, so hybrid is the better choice.

Evaluating multiple approaches and using hybrid search also covers the
hybrid-search best-practice.

---

## LLM evaluation

Using the `llm_judge.py` script, I compared 3 prompt styles (concise, detailed,
and baseline) using an LLM as a judge over 50 questions.

```
variant       mean_score  RELEVANT  PARTLY   NON
------------------------------------------------
concise            0.960        48       0     2
detailed           0.960        48       0     2
baseline           0.940        47       0     3
```

Based on these results I decided to use the **detailed** variant as my default in
the `INSTRUCTIONS` block in `rag.py`. Concise and detailed tied on score, and I
went with detailed because the richer, stat-backed answers fit a sports assistant
better.

### The judge bug I had to fix first

Before I could trust those numbers, I had to fix my `llm_judge.py` script. When I
first ran it, I got these much worse results:

```
variant       mean_score  RELEVANT  PARTLY   NON
------------------------------------------------
baseline           0.720        32       8    10
detailed           0.630        30       3    17
concise            0.570        23      11    16
```

The problem was that the judge model's training cutoff is before the 2026 World
Cup. It was classifying correct answers as non-relevant (for example, an answer
about Messi scoring goals) because (1) it believed Messi retired from
international play in 2021, and (2) it thought the 2026 World Cup hadn't been
played yet. So it was "fact-checking" my answers against its own outdated
knowledge instead of the data.

To fix this, I changed the `JUDGE_PROMPT` to judge answers **only** against the
CONTEXT provided (the tournament data) and to treat that context as the sole
source of truth. After that fix, the false non-relevants disappeared and the
scores jumped to the numbers above, which reflect the system's real answer
quality.

---

## Interface

The interface is a **Streamlit** chat app (`app.py`). You can ask questions in
natural language, see the conversation history, expand a "Sources" panel to see
which documents were retrieved, switch the retrieval method (hybrid/vector/text)
from the sidebar, and leave thumbs up/down feedback on any answer. The knowledge
base is built once at startup and cached, so questions after the first are fast.

---

## Ingestion pipeline

Ingestion is automated with **dlt** (`pipeline.py`). In one run it:

1. Loads the raw dataset CSVs into DuckDB as normalized tables (excluding the
   prediction files),
2. Builds the ~1,400 knowledge-base documents and loads them into DuckDB, and
3. Refreshes the vector embeddings.

Because dlt handles the schema inference, typing, and load tracking, it's a proper
tool-managed pipeline rather than an ad-hoc script.

---

## Monitoring

Every question/answer is logged to **Postgres** (`db.py`) with its tokens, cost,
response time, and retrieval method, and users can leave thumbs up/down feedback.
**Grafana** reads these tables and renders an auto-provisioned dashboard with 11
panels, including:

- Total questions, total cost, average response time, feedback score
- Questions / response time / token usage / cost over time
- Feedback breakdown (thumbs up vs down)
- Retrieval method usage
- A table of recent questions

---

## Containerization

Everything runs from a single `docker compose up --build`:

- **app**: the Streamlit application (built from the `Dockerfile`)
- **postgres**: the monitoring database
- **grafana**: the dashboard

The app container connects to Postgres by service name, auto-downloads the
embedding model and builds the knowledge base on first launch, and persists the
knowledge base in a named volume.

---

## Known limitations and design decisions

**Aggregation and high-level questions.** RAG retrieves only the top few documents
per query, so it struggles with questions whose answer requires scanning the whole
dataset, like *"which team scored the most goals?"* or *"who won the World Cup?"*
(no single match or player document explicitly says who won the tournament). To
handle this, the ingestion pipeline pre-computes tournament-wide aggregates and
stores each as its own summary document (team goal rankings, all 12 group
standings, top scorers/assists, clean sheets, biggest wins, venues by matches
hosted, the tournament champion, and the full knockout bracket). That way a single
retrieved summary document can answer the whole class of questions that
single-document retrieval otherwise misses.

**Data accuracy.** The assistant faithfully reports whatever is in the dataset. The
Kaggle dataset is community-maintained, so if a record is wrong, the assistant will
repeat it. This is correct behavior for a grounded RAG system (it answers from the
data, not from outside knowledge), but it means answer accuracy depends on the
dataset's accuracy.

---

## Possible improvements

- **Document re-ranking** after retrieval, to push the single best document to the
  top before it hits the LLM.
- **Query rewriting** to expand short questions into retrieval-friendly queries.
- **Scheduled ingestion** (for example, Kestra in the compose stack) to refresh the
  data on a cadence instead of manually.
- **Strict LLM Rules** Since I told the LLM to pull only from the data to answer questions, it doesn't use
  additional sources from the internet to augment/help if it doesn't find an explicit answer.