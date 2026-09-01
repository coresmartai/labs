"""FastAPI surface - OrderBot endpoints."""
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

from app.config import get_settings
from app.schemas import LookupRequest, LookupResponse
from app.llm import call_with_tools, LoopExceededError, SchemaCorrectionExhausted

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("orderbot")

# Map provider ID → pinned model string
_PROVIDER_MODEL: dict[str, str] = {
    "openai": settings.model_name,
    "nano":   settings.fast_model_name,
}

app = FastAPI(
    title="OrderBot",
    version="0.1.0",
    description="Week 3 - bounded tool loop with retry-with-correction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def serve_ui():
    ui = Path(__file__).parent.parent / "index.html"
    if not ui.exists():
        raise HTTPException(404, detail="index.html not found")
    return FileResponse(ui)


@app.post("/lookup", response_model=LookupResponse)
def lookup(req: LookupRequest):
    model_name = _PROVIDER_MODEL[req.provider]
    try:
        result = call_with_tools(req.message, model_name=model_name)
        result["provider"] = req.provider
        result["model"] = model_name
        return JSONResponse(
            content=result,
            headers={
                "X-Tool-Call-Count": str(result["tool_call_count"]),
                "X-Retry-Count":     str(result["retry_count"]),
                "X-Latency-Ms":      str(result["latency_ms"]),
            },
        )
    except LoopExceededError:
        raise HTTPException(429, detail="Tool loop exceeded; retry later.")
    except SchemaCorrectionExhausted:
        raise HTTPException(
            502,
            detail="Could not extract a valid order request from your message.",
        )


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "model": s.model_name,
        "models": {
            "openai": s.model_name,
            "nano":   s.fast_model_name,
        },
    }


@app.get("/readme", include_in_schema=False)
def serve_readme():
    import markdown as _md
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="UTF-8"><title>OrderBot - README</title>'
        "<style>"
        "body{background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif;"
        "max-width:860px;margin:40px auto;padding:0 20px;line-height:1.7}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px}"
        "code{background:#21262d;border-radius:4px;padding:2px 6px;font-family:monospace}"
        "pre{background:#161b22;border:1px solid #30363d;border-radius:8px;"
        "padding:16px;overflow-x:auto}"
        "a{color:#58a6ff}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #30363d;padding:8px 12px}"
        "th{background:#161b22}"
        "blockquote{border-left:3px solid #30363d;margin:0;padding-left:16px;color:#7d8590}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")
