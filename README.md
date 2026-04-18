# Grabpic

**Intelligent Identity & Retrieval Engine** — a FastAPI backend that ingests event photos from a storage folder, detects multiple faces per image, assigns a stable internal **`grab_id`** per unique identity (via DeepFace embeddings + cosine similarity), persists **image ↔ many grab_ids** in **PostgreSQL**, and exposes **selfie-as-key** authentication so a returned `grab_id` authorizes gallery retrieval.

Interactive API docs (Swagger UI): `http://localhost:8000/docs` (OpenAPI: `/openapi.json`).

## Architecture (high level)

**Static diagram** (white background — renders on GitHub, IDEs, and PDFs):

![Grabpic system architecture](docs/architecture.svg)

**Same architecture as Mermaid** (renders as a diagram on [github.com](https://github.com/azzammasood/grabpic); in some offline Markdown previews it may appear as a code block — use the SVG above, or open the file on GitHub):

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ffffff','primaryTextColor':'#0f172a','primaryBorderColor':'#1e293b','lineColor':'#475569','secondaryColor':'#f1f5f9','tertiaryColor':'#ffffff','background':'#ffffff','mainBkg':'#ffffff','secondBkg':'#f8fafc'}}}%%
flowchart TB
  subgraph clients[Clients]
    C[Swagger / curl / Postman]
  end
  subgraph vol[Storage volume]
    S[RAW images]
  end
  subgraph api[FastAPI + Uvicorn]
    I[Ingest crawl + upload]
    A[Selfie auth]
    G[Gallery endpoints]
  end
  subgraph ml[DeepFace lazy-loaded]
    F[Facenet512 + cosine match]
  end
  subgraph db[PostgreSQL]
    GS[(grab_subjects)]
    SI[(stored_images)]
    MAP[(image_grab_map)]
  end
  C --> I
  C --> A
  C --> G
  S --> I
  I --> F
  A --> F
  F --> GS
  I --> SI
  SI --> MAP
  GS --> MAP
  G --> SI
  G --> MAP
```

### Relational schema

| Table | Purpose |
|-------|---------|
| `grab_subjects` | One row per enrolled identity: `grab_id` (UUID) + `face_encoding` (float array, 512-d Facenet512, L2-normalized at write time). |
| `stored_images` | One row per ingested file path (relative to `STORAGE_PATH`) + optional SHA-256. |
| `image_grab_map` | **Many-to-many** link: `(image_id, grab_id)` composite PK — one image can reference many faces. |

## Assumptions & design choices

1. **Storage crawl**: The “crawler” walks `STORAGE_PATH` for `jpg/jpeg/png/webp/bmp` (optionally recursive). It is a filesystem scan, not S3-specific; mounting object storage as a volume is assumed for cloud deployments.
2. **Identity clustering**: There is **one canonical embedding per `grab_id`**. New faces are matched against existing subjects with **cosine similarity** ≥ `MATCH_THRESHOLD`; otherwise a new `grab_id` is created. (Production systems often maintain multiple prototypes per identity; this is intentionally simple for the hackathon scope.)
3. **Multi-face images**: Each detected face in an image is processed independently and linked to the same `stored_images` row via `image_grab_map`.
4. **Selfie authentication**: The uploaded “search token” may contain multiple people; the **largest detected face** (by bounding-box area) is used for matching.
5. **Authorization model**: After a successful selfie, the client uses `Authorization: Bearer <grab_id>` to prove possession of that identity. The token must equal the `grab_id` in the URL for `GET /v1/grabs/{grab_id}/images` (defense-in-depth against casual IDOR).
6. **First run / models**: DeepFace downloads model weights on **first face-related request** (ingest or selfie), not on `/health` — the API process starts quickly; the first crawl/upload/selfie may take several minutes while TensorFlow/DeepFace initialize (network required once for weights).
7. **Performance**: Embedding search is linear in the number of enrolled subjects — adequate for demo scale (hundreds / low thousands). At true marathon scale you would add ANN (e.g. pgvector / FAISS) and batching.

## Prerequisites

- **Docker** + **Docker Compose** (recommended), or Python 3.11+ with PostgreSQL 16+ and sufficient libs for OpenCV/TensorFlow (Linux is the smoothest path).

## Quick start (Docker Compose)

**Requires Docker Desktop (or another Docker engine) running.** From the repository root:

```bash
docker compose up --build
```

The `./storage` folder on your machine is mounted into the API container at `/data/storage` (see `STORAGE_PATH` in `docker-compose.yml`), so you can drop images into `storage/` on the host and crawl them immediately.

- API: `http://localhost:8000`
- Postgres: `localhost:5432` (user/password/db: `grabpic` / `grabpic` / `grabpic`)

Copy `.env.example` to `.env` only if you run the API **outside** Docker and need to point at a local DB.

### Why `docker compose build` sometimes re-downloads everything

Docker caches **image layers**. The `RUN pip install ...` step only stays cached if **every layer above it is unchanged**. Typical reasons you see a full pip download again:

- **`requirements.txt` changed** (any line) — invalidates that `COPY` and every step after it, including `pip install`.
- **`Dockerfile` changed** above the pip step (e.g. `apt-get`, base image).
- **You passed `--no-cache` / BuildKit “no cache”** — forces a clean rebuild.
- **Base image was re-pulled** — `python:3.11-slim-bookworm` can point at a newer digest; the first layer changes, so downstream layers rebuild.

The Dockerfile uses `pip install --no-cache-dir`, which avoids bloating the image with pip’s *local* wheel cache inside the layer; wheels are still downloaded whenever that layer is rebuilt. To rebuild only when code changes, keep `requirements.txt` stable and rely on normal `docker compose build` (without `--no-cache`).

### Manual testing guide

Validation is **manual** (Swagger UI, `curl`, or Postman) so you are not blocked by slow first-time ML downloads in a scripted health loop.

1. **Start stack** (from repo root):

   ```bash
   docker compose up --build
   ```

   Wait until `docker compose ps` shows the **api** service as **healthy** (Compose runs an HTTP check on `/health`). If it stays “starting”, watch logs:

   ```bash
   docker compose logs -f api
   ```

2. **Open docs**: [http://localhost:8000/docs](http://localhost:8000/docs) — try `GET /health` (should be immediate).

3. **Ingest faces** — either:

   - Put JPEG/PNG files under `./storage` on the host, then `POST /v1/ingest/crawl` with body `{"recursive": true}`, **or**
   - `POST /v1/ingest/image` with a form file (`curl -F "file=@path/to/photo.jpg" ...`).

   The **first** ingest/selfie after a cold start can take **several minutes** while DeepFace downloads model weights; later calls are much faster.

4. **Selfie auth**: `POST /v1/auth/selfie` with the same or a similar face image; copy `grab_id` from the JSON.

5. **Gallery**: `GET /v1/me/images` with header `Authorization: Bearer <grab_id>`, or `GET /v1/grabs/{grab_id}/images` with the same Bearer.

#### Why an old “wait for /health then run pytest” script could time out

That pattern is **not** slow because of “automated testing” — it fails if the script’s wait window is shorter than **process + import + bind** time. Previously, importing the app pulled TensorFlow at import time, so **nothing listened on port 8000 until ML stacks finished loading**. DeepFace is now **lazy-imported** on the first face operation, so `/health` should respond quickly; face endpoints still pay the one-time cost on first use.

## Local development (without Docker for the API)

1. Create a PostgreSQL database and set `DATABASE_URL` (see `.env.example`).
2. `python -m venv .venv` then `pip install -r requirements.txt`
3. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

On Windows, installing TensorFlow/OpenCV stacks can be fragile; prefer Docker or WSL2 for a predictable environment.

**Completing `pip install -r requirements.txt` on Windows:** if resolution hangs or conflicts, rely on the **Dockerfile** instead — `docker compose build` installs the same stack inside Linux and is the supported path for a full environment.

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://grabpic:grabpic@localhost:5432/grabpic` | SQLAlchemy Postgres URL |
| `STORAGE_PATH` | `./storage` | Root folder crawled by `/v1/ingest/crawl` |
| `FACE_MODEL` | `Facenet512` | DeepFace model (keep consistent for ingest + auth) |
| `DETECTOR_BACKEND` | `retinaface` | Face detector; falls back to OpenCV on failure |
| `MATCH_THRESHOLD` | `0.42` | Minimum cosine similarity to treat embeddings as same person |

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness |
| `POST` | `/v1/ingest/crawl` | Scan `STORAGE_PATH` and index faces |
| `POST` | `/v1/ingest/image` | Upload a single image file |
| `POST` | `/v1/auth/selfie` | Selfie-as-key → returns `grab_id` + similarity |
| `GET` | `/v1/grabs/{grab_id}/images` | List images for a subject (Bearer must match path) |
| `GET` | `/v1/me/images` | Same as above using only `Authorization: Bearer <grab_id>` |

### Example `curl` flows

Place some photos under `./storage` (or rely on the Compose volume at `/data/storage` inside the container). Then:

**1) Crawl the storage folder**

```bash
curl -s -X POST "http://localhost:8000/v1/ingest/crawl" \
  -H "Content-Type: application/json" \
  -d "{\"recursive\": true}" | jq
```

**2) Upload one image**

```bash
curl -s -X POST "http://localhost:8000/v1/ingest/image" \
  -F "file=@/path/to/photo.jpg" | jq
```

**3) Authenticate with a selfie (search token)**

```bash
curl -s -X POST "http://localhost:8000/v1/auth/selfie" \
  -F "file=@/path/to/selfie.jpg" | jq
```

Save the returned `grab_id` as `GID`.

**4) Fetch that user’s images**

```bash
curl -s "http://localhost:8000/v1/grabs/$GID/images" \
  -H "Authorization: Bearer $GID" | jq
```

Equivalent convenience endpoint:

```bash
curl -s "http://localhost:8000/v1/me/images" \
  -H "Authorization: Bearer $GID" | jq
```

### Error shape

Most errors return JSON like:

```json
{"detail": {"code": "unknown_face", "message": "..."}}
```

Validation errors use FastAPI’s default `422` structure.

## Tech stack

- **Python 3.11**, **FastAPI**, **Uvicorn**
- **PostgreSQL** + **SQLAlchemy 2**
- **DeepFace** (Facenet512 embeddings) + **NumPy** for similarity

## License

Provided as a hackathon submission; adjust licensing as needed for your team.
