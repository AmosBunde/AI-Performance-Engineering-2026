# GitHub Repository Summarizer API

A FastAPI service that accepts a GitHub repository URL and returns a human-readable summary of what the project does, the technologies it uses, and how it is structured.

---

## Setup & Run

### Prerequisites
- Python 3.10+
- An LLM API key (Nebius Token Factory **or** Anthropic)

### 1. Clone / unzip the project

```bash
cd github-summarizer   # or wherever you extracted the files
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

**Option A – Nebius Token Factory (primary)**
```bash
export NEBIUS_API_KEY="your-nebius-key-here"
```

**Option B – Anthropic**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

> The service checks for `ANTHROPIC_API_KEY` first; if absent it falls back to `NEBIUS_API_KEY`.

*Optional – GitHub Token (raises API rate limit from 60 → 5000 req/hr)*
```bash
export GITHUB_TOKEN="your-github-pat"
```

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Using the API

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

### Example response

```json
{
  "summary": "**Requests** is a popular Python HTTP library that simplifies sending HTTP/1.1 requests. It abstracts away complexity, offering a human-friendly API for making GET, POST, and other HTTP calls with minimal boilerplate.",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer", "idna"],
  "structure": "The project follows a standard Python package layout: source code lives in `src/requests/`, tests in `tests/`, and documentation in `docs/`. Configuration is managed via `pyproject.toml` and `setup.cfg`."
}
```

### Error response

```json
{
  "status": "error",
  "message": "Repository not found or is private."
}
```

---

## Model choice

| Key set | Model used | Why |
|---|---|---|
| `NEBIUS_API_KEY` | `deepseek-ai/DeepSeek-V3` | State-of-the-art open model available on Nebius with a large context window; excellent at code understanding and JSON generation. |
| `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | Extremely fast and cost-efficient; great at following structured JSON output instructions reliably. |

---

## Strategy for handling repository contents

### What we include
1. **Full directory tree** – always included (text only, very cheap). Gives the LLM the project shape at a glance.
2. **High-value files first** – `README`, `package.json`, `pyproject.toml`, `Dockerfile`, `go.mod`, `requirements.txt`, etc. These files reveal purpose and dependencies with minimal tokens.
3. **Additional source files** – up to a combined budget of ~40 000 characters of other non-skipped files to fill in implementation detail.

### What we skip
- **Binary & media files** – `.png`, `.jpg`, `.pdf`, `.so`, `.exe`, etc.
- **Lock files** – `package-lock.json`, `yarn.lock`, `poetry.lock`, etc. (huge, zero signal).
- **Generated / dependency directories** – `node_modules/`, `__pycache__/`, `.venv/`, `dist/`, `build/`, `vendor/`, etc.
- **Hidden directories** – anything starting with `.` (except config files at root).

### Why this approach
- A README + key config files typically give 80 % of the signal needed to understand a project.
- Sending the full tree as text (not fetching every file) lets the LLM infer structure without hundreds of API calls.
- Per-file truncation (3 000 chars) and a total budget (40 000 chars) keep us well within context limits while still providing meaningful source snippets.
