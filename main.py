import os
import re
import base64
import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="GitHub Repo Summarizer")

# ── Config ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEBIUS_API_KEY    = os.getenv("NEBIUS_API_KEY", "")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")   # optional, increases rate limits

# Files/dirs to skip (binaries, lock files, generated assets, etc.)
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", "site-packages", "vendor", "target", "out",
}
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg", ".so", ".dylib",
    ".dll", ".exe", ".bin", ".lock", ".min.js", ".min.css",
    ".map", ".snap", ".pyc",
}
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    "pnpm-lock.yaml", "composer.lock", "Gemfile.lock", "cargo.lock",
}
# High-value files to always include if present
HIGH_VALUE = {
    "readme.md", "readme.rst", "readme.txt", "readme",
    "pyproject.toml", "setup.py", "setup.cfg", "package.json",
    "cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", ".env.example", "requirements.txt",
    "gemfile", "composer.json",
}

MAX_FILE_CHARS   = 3_000   # per file
MAX_TOTAL_CHARS  = 40_000  # total context budget


# ── Request / Response models ────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    github_url: str

class SummarizeResponse(BaseModel):
    summary: str
    technologies: list[str]
    structure: str


# ── GitHub helpers ───────────────────────────────────────────────────────────
def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL."""
    m = re.search(r"github\.com/([^/]+)/([^/?\#]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2).rstrip(".git")


def gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


async def get_tree(owner: str, repo: str) -> list[dict]:
    """Fetch the full recursive file tree via Git Trees API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=gh_headers())
        r.raise_for_status()
        data = r.json()
    return data.get("tree", [])


async def fetch_file(owner: str, repo: str, path: str) -> Optional[str]:
    """Fetch a single file's content (decoded text). Returns None on failure."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=gh_headers())
        if r.status_code != 200:
            return None
        data = r.json()
    if data.get("encoding") == "base64":
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return None
    return data.get("content")


def should_skip(path: str) -> bool:
    """True if this file/path should be excluded."""
    parts = path.split("/")
    # Skip if any dir component is in SKIP_DIRS
    for p in parts[:-1]:
        if p in SKIP_DIRS or p.startswith("."):
            return True
    filename = parts[-1].lower()
    if filename in SKIP_FILENAMES:
        return True
    _, ext = os.path.splitext(filename)
    if ext in SKIP_EXTENSIONS:
        return True
    return False


def is_high_value(path: str) -> bool:
    return path.lower().split("/")[-1] in HIGH_VALUE


def build_tree_string(blobs: list[str]) -> str:
    """Create a compact directory-tree text from a list of file paths."""
    lines = []
    for p in sorted(blobs):
        depth = p.count("/")
        name  = p.split("/")[-1]
        lines.append("  " * depth + name)
    return "\n".join(lines)


# ── LLM call ────────────────────────────────────────────────────────────────
async def call_llm(prompt: str) -> str:
    """
    Call Anthropic claude-3-5-haiku if ANTHROPIC_API_KEY is set,
    otherwise fall back to Nebius (DeepSeek-V3) using NEBIUS_API_KEY.
    """
    if ANTHROPIC_API_KEY:
        return await _call_anthropic(prompt)
    if NEBIUS_API_KEY:
        return await _call_nebius(prompt)
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or NEBIUS_API_KEY."
    )


async def _call_anthropic(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
    return data["content"][0]["text"]


async def _call_nebius(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.tokenfactory.nebius.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NEBIUS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-ai/DeepSeek-V3-0324-fast",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def build_prompt(owner: str, repo: str, tree_str: str, file_contents: dict[str, str]) -> str:
    files_section = ""
    for path, content in file_contents.items():
        snippet = content[:MAX_FILE_CHARS]
        if len(content) > MAX_FILE_CHARS:
            snippet += "\n... [truncated]"
        files_section += f"\n\n### {path}\n```\n{snippet}\n```"

    return f"""You are a technical analyst. Analyze the GitHub repository `{owner}/{repo}` based on the information below and respond with ONLY valid JSON — no markdown fences, no extra text.

## Directory tree
```
{tree_str}
```

## Key file contents
{files_section}

Respond with this exact JSON structure:
{{
  "summary": "<2-4 sentence human-readable description of what this project does and its purpose>",
  "technologies": ["<tech1>", "<tech2>", "..."],
  "structure": "<1-3 sentence description of the project layout and organisation>"
}}
"""


# ── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    try:
        owner, repo = parse_github_url(req.github_url)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    # 1. Fetch file tree
    try:
        tree = await get_tree(owner, repo)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 404:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Repository not found or is private."})
        return JSONResponse(status_code=502, content={"status": "error", "message": f"GitHub API error: {status}"})
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "message": f"Failed to fetch repository tree: {e}"})

    blobs = [item["path"] for item in tree if item["type"] == "blob" and not should_skip(item["path"])]

    # 2. Build directory tree string (all non-skipped paths)
    tree_str = build_tree_string(blobs)

    # 3. Prioritise files to fetch
    high = [p for p in blobs if is_high_value(p)]
    others = [p for p in blobs if not is_high_value(p)]

    # Fetch high-value files first, then fill budget with others
    to_fetch = high + others[:60]   # cap total API calls

    file_contents: dict[str, str] = {}
    total_chars = 0

    for path in to_fetch:
        if total_chars >= MAX_TOTAL_CHARS:
            break
        text = await fetch_file(owner, repo, path)
        if text is None:
            continue
        # Skip binary-looking content
        if "\x00" in text[:512]:
            continue
        snippet = text[:MAX_FILE_CHARS]
        file_contents[path] = snippet
        total_chars += len(snippet)

    # 4. Build prompt & call LLM
    prompt = build_prompt(owner, repo, tree_str, file_contents)

    try:
        raw = await call_llm(prompt)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "message": f"LLM call failed: {e}"})

    # 5. Parse JSON response
    import json
    # Strip possible markdown fences just in case
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse(status_code=502, content={"status": "error", "message": f"LLM returned invalid JSON: {raw[:300]}"})

    return SummarizeResponse(
        summary=result.get("summary", ""),
        technologies=result.get("technologies", []),
        structure=result.get("structure", ""),
    )


@app.get("/")
async def root():
    return {"message": "GitHub Summarizer API. POST /summarize with {\"github_url\": \"...\"}"}
