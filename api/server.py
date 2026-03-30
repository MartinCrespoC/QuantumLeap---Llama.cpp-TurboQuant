"""
TurboQuant Server — Full-featured LLM platform
Ollama-compatible API + OpenAI API + Model Manager + Benchmarks + Web UI
"""

import asyncio
import json
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PROJECT_ROOT / "engine" / "llama.cpp"
MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"
BENCH_DIR = PROJECT_ROOT / "benchmarks"
LLAMA_SERVER_BIN = ENGINE_DIR / "build" / "bin" / "llama-server"
LLAMA_BENCH_BIN = ENGINE_DIR / "build" / "bin" / "llama-bench"
CONFIG_FILE = PROJECT_ROOT / "config.json"

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8081"))
API_PORT = int(os.environ.get("API_PORT", "11434"))
DEFAULT_GPU_LAYERS = int(os.environ.get("GPU_LAYERS", "99"))
DEFAULT_CTX = int(os.environ.get("CTX_SIZE", "4096"))

HF_API = "https://huggingface.co/api"

# ─── State ─────────────────────────────────────────────────────────────────────

llama_process: Optional[subprocess.Popen] = None
current_model: Optional[str] = None
current_model_path: Optional[str] = None
model_load_time: float = 0.0
active_downloads: dict[str, dict] = {}
benchmark_results: list[dict] = []


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"default_model": None, "gpu_layers": DEFAULT_GPU_LAYERS, "ctx_size": DEFAULT_CTX}


def _save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ─── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    yield
    await _stop_llama_server()


app = FastAPI(title="TurboQuant", version="QuantumLeap v0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Model Utilities ──────────────────────────────────────────────────────────

def find_models() -> list[dict]:
    models = []
    if not MODELS_DIR.exists():
        return models
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        size = f.stat().st_size
        name = f.stem
        models.append({
            "name": name, "model": name,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)),
            "size": size, "path": str(f),
            "details": {
                "format": "gguf", "family": _guess_family(name),
                "parameter_size": _guess_params(name),
                "quantization_level": _guess_quant(name),
            },
        })
    return models


def _guess_quant(name: str) -> str:
    nl = name.lower()
    for q in ["iq2_xxs","iq2_xs","iq2_s","iq2_m","iq3_xxs","iq3_xs","iq3_s","iq4_xs","iq4_nl",
              "tq1_0","tq2_0","q2_k","q2_k_s","q3_k_s","q3_k_m","q3_k_l",
              "q4_0","q4_1","q4_k_s","q4_k_m","q5_0","q5_1","q5_k_s","q5_k_m",
              "q6_k","q8_0","f16","f32","bf16"]:
        if q in nl:
            return q.upper()
    return "unknown"


def _guess_family(name: str) -> str:
    nl = name.lower()
    for f in ["llama","mistral","qwen","phi","gemma","smollm","deepseek","yi","command","starcoder","codellama"]:
        if f in nl:
            return f.capitalize()
    return "Unknown"


def _guess_params(name: str) -> str:
    m = re.search(r'(\d+\.?\d*)[bB]', name)
    return f"{m.group(1)}B" if m else "unknown"


def _resolve_model_path(model_name: str) -> Optional[Path]:
    if not MODELS_DIR.exists():
        return None
    exact = MODELS_DIR / f"{model_name}.gguf"
    if exact.exists():
        return exact
    for f in MODELS_DIR.rglob("*.gguf"):
        if model_name.lower() in f.stem.lower():
            return f
    return None


# ─── Hardware Detection ───────────────────────────────────────────────────────

# Bytes per parameter for each quantization type
QUANT_BPP: dict[str, float] = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0,
    "Q8_0": 1.1, "Q6_K": 0.83, "Q5_K_M": 0.69, "Q5_K_S": 0.69,
    "Q5_0": 0.69, "Q5_1": 0.69,
    "Q4_K_M": 0.56, "Q4_K_S": 0.56, "Q4_0": 0.55, "Q4_1": 0.6,
    "Q3_K_M": 0.44, "Q3_K_S": 0.43, "Q3_K_L": 0.45,
    "Q2_K": 0.34, "Q2_K_S": 0.34,
    "IQ4_XS": 0.52, "IQ4_NL": 0.55,
    "IQ3_XXS": 0.40, "IQ3_XS": 0.41, "IQ3_S": 0.42,
    "IQ2_XXS": 0.31, "IQ2_XS": 0.32, "IQ2_S": 0.33, "IQ2_M": 0.33,
    "TQ1_0": 0.22, "TQ2_0": 0.30,
}


def _detect_vram_mb() -> tuple[float, float]:
    """Detect available GPU VRAM in MB using nvidia-smi. Returns (total, free)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("\n")[0].split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    return 4096.0, 3500.0  # Default RTX 3050


def _detect_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return "Unknown GPU"


def _guess_params_float(name: str) -> float:
    """Extract parameter count as float from model name. Returns 0 if unknown."""
    m = re.search(r'(\d+\.?\d*)[bB]', name)
    return float(m.group(1)) if m else 0.0


def _estimate_layers(params_b: float) -> int:
    """Estimate number of transformer layers from parameter count."""
    if params_b <= 0:
        return 32
    if params_b <= 1:
        return 16
    if params_b <= 2:
        return 24
    if params_b <= 4:
        return 28
    if params_b <= 10:
        return 32
    if params_b <= 15:
        return 40
    if params_b <= 35:
        return 48
    if params_b <= 75:
        return 80
    return 96


def calculate_optimal_ngl(model_path: Path, vram_free_mb: float, ctx_size: int) -> int:
    """Calculate optimal number of GPU layers to avoid OOM."""
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    params_b = _guess_params_float(model_path.stem)
    total_layers = _estimate_layers(params_b)

    # Reserve VRAM for KV cache + overhead
    kv_cache_mb = (ctx_size / 1024) * 400  # ~400MB per 1K tokens with q4_0 cache
    overhead_mb = 400
    available_mb = vram_free_mb - kv_cache_mb - overhead_mb

    if available_mb <= 0:
        return 0  # CPU only

    layer_size_mb = model_size_mb / total_layers
    gpu_layers = int(available_mb / layer_size_mb)

    # Safety margin: use 85% of calculated
    gpu_layers = int(gpu_layers * 0.85)
    return max(0, min(gpu_layers, total_layers + 1))


def calculate_model_fit(params_b: float, quant: str, vram_total_mb: float = 4096,
                        ram_total_mb: float = 24576) -> dict:
    """Calculate if a model fits in the hardware and how."""
    bpp = QUANT_BPP.get(quant.upper(), 0.56)
    model_size_mb = params_b * bpp * 1024  # GB -> MB
    kv_mb = 800  # ~800MB for 4K ctx with q4_0 KV cache
    overhead_mb = 400
    total_needed_mb = model_size_mb + kv_mb + overhead_mb

    if total_needed_mb <= vram_total_mb:
        return {"fits": "gpu", "model_size_gb": round(model_size_mb / 1024, 1),
                "vram_needed_gb": round(total_needed_mb / 1024, 1), "recommended": True}
    elif model_size_mb <= vram_total_mb + ram_total_mb:
        gpu_pct = int(vram_total_mb / total_needed_mb * 100)
        return {"fits": "mixed", "model_size_gb": round(model_size_mb / 1024, 1),
                "vram_needed_gb": round(total_needed_mb / 1024, 1), "gpu_pct": gpu_pct}
    elif model_size_mb <= ram_total_mb:
        return {"fits": "cpu", "model_size_gb": round(model_size_mb / 1024, 1),
                "vram_needed_gb": round(total_needed_mb / 1024, 1)}
    else:
        return {"fits": "no", "model_size_gb": round(model_size_mb / 1024, 1),
                "vram_needed_gb": round(total_needed_mb / 1024, 1)}


def _generate_compatibility_table(vram_mb: float, ram_mb: float) -> list[dict]:
    """Generate a compatibility table for common model sizes."""
    sizes = [1.7, 3, 7, 8, 13, 27, 70]
    quants = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XXS"]
    table = []
    for size in sizes:
        row = {"params_b": size, "quants": {}}
        for q in quants:
            row["quants"][q] = calculate_model_fit(size, q, vram_mb, ram_mb)
        table.append(row)
    return table


# ─── Engine Management ────────────────────────────────────────────────────────

async def _start_llama_server(model_path: Path, gpu_layers: int = DEFAULT_GPU_LAYERS,
                               ctx_size: int = DEFAULT_CTX) -> bool:
    global llama_process, current_model, current_model_path, model_load_time
    await _stop_llama_server()

    bin_path = LLAMA_SERVER_BIN
    if not bin_path.exists():
        alt = ENGINE_DIR / "build" / "llama-server"
        if alt.exists():
            bin_path = alt
        else:
            raise HTTPException(500, f"llama-server not found at {LLAMA_SERVER_BIN}")

    env = os.environ.copy()
    env["PATH"] = f"/opt/cuda/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/opt/cuda/lib64:/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    # Auto-calculate GPU layers if set to 99 (auto) and model is too large
    use_uma = False
    use_mlock = True  # Always enable mlock for consistent RAM access
    if gpu_layers >= 99:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        _, vram_free = _detect_vram_mb()
        kv_est = (ctx_size / 1024) * 400
        if model_size_mb + kv_est + 400 > vram_free:
            gpu_layers = calculate_optimal_ngl(model_path, vram_free, ctx_size)
            # Enable UMA for mixed offloading — gives ~10% speed boost
            # by allowing GPU to access overflow memory transparently
            if gpu_layers > 0 and model_size_mb > vram_free * 0.8:
                use_uma = True
                # With UMA we can safely push ~50% more layers to GPU
                gpu_layers = min(int(gpu_layers * 1.5), _estimate_layers(_guess_params_float(model_path.stem)) + 1)
            print(f"[QuantumLeap] Model {model_path.stem}: {model_size_mb:.0f}MB, "
                  f"VRAM free: {vram_free:.0f}MB → auto ngl={gpu_layers}"
                  f"{' +UMA' if use_uma else ''}")

    cmd = [str(bin_path), "-m", str(model_path), "--host", "127.0.0.1",
           "--port", str(LLAMA_PORT), "-ngl", str(gpu_layers), "-c", str(ctx_size),
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--metrics"]

    if use_mlock:
        cmd.append("--mlock")

    # MoE models benefit from --no-mmap (pre-load to RAM vs memory-mapping)
    # Gives +28% speed boost due to scattered expert access patterns
    if is_moe:
        cmd.append("--no-mmap")

    if use_uma:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

    print(f"[TurboQuant] Starting: {' '.join(cmd[-10:])}")
    llama_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    current_model = model_path.stem
    current_model_path = str(model_path)
    model_load_time = time.time()

    for _ in range(120):
        await asyncio.sleep(0.5)
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2)
                if r.status_code == 200:
                    return True
        except Exception:
            if llama_process.poll() is not None:
                stderr = llama_process.stderr.read().decode() if llama_process.stderr else ""
                raise HTTPException(500, f"llama-server crashed: {stderr[-500:]}")
    raise HTTPException(500, "llama-server failed to start within 60s")


async def _stop_llama_server():
    global llama_process, current_model, current_model_path
    if llama_process and llama_process.poll() is None:
        llama_process.terminate()
        try:
            llama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            llama_process.kill()
    llama_process = None
    current_model = None
    current_model_path = None


async def _ensure_model(model_name: str):
    if current_model and model_name.lower() in current_model.lower():
        return
    path = _resolve_model_path(model_name)
    if not path:
        raise HTTPException(404, f"Model '{model_name}' not found in {MODELS_DIR}")
    cfg = _load_config()
    await _start_llama_server(path, cfg.get("gpu_layers", DEFAULT_GPU_LAYERS),
                               cfg.get("ctx_size", DEFAULT_CTX))


# ─── Streaming Helpers ────────────────────────────────────────────────────────

async def _proxy_stream(model: str, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
    await _ensure_model(model)
    payload = {"model": model, "messages": messages, "stream": True,
               **{k: v for k, v in kwargs.items() if v is not None}}
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                                  json=payload, timeout=300) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content") or ""
                            finish = data.get("choices", [{}])[0].get("finish_reason")
                            ollama_chunk = {
                                "model": model,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "message": {"role": "assistant", "content": content},
                                "done": finish is not None,
                            }
                            if finish:
                                ollama_chunk["done_reason"] = "stop"
                                usage = data.get("usage") or {}
                                ollama_chunk["prompt_eval_count"] = usage.get("prompt_tokens", 0)
                                ollama_chunk["eval_count"] = usage.get("completion_tokens", 0)
                            yield json.dumps(ollama_chunk) + "\n"
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue


async def _generate_nonstream(model: str, messages: list[dict], **kwargs) -> dict:
    await _ensure_model(model)
    payload = {"model": model, "messages": messages, "stream": False,
               **{k: v for k, v in kwargs.items() if v is not None}}
    async with httpx.AsyncClient() as client:
        r = await client.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                               json=payload, timeout=300)
        data = r.json()
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage") or {}
    return {
        "model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": choice.get("message", {"role": "assistant", "content": ""}),
        "done": True, "done_reason": "stop",
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
        "total_duration": int((time.time() - model_load_time) * 1e9),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA-COMPATIBLE API
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = True
    options: Optional[dict] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    system: Optional[str] = None
    stream: bool = True
    options: Optional[dict] = None


@app.get("/")
async def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>TurboQuant</h1><p>Web UI not found</p>")


@app.get("/api/tags")
async def list_tags():
    return {"models": find_models()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    opts = req.options or {}
    kw = dict(temperature=opts.get("temperature"), top_p=opts.get("top_p"),
              max_tokens=opts.get("num_predict"))
    if req.stream:
        return StreamingResponse(_proxy_stream(req.model, req.messages, **kw),
                                  media_type="application/x-ndjson")
    result = await _generate_nonstream(req.model, req.messages, **kw)
    return JSONResponse(result)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})
    opts = req.options or {}
    kw = dict(temperature=opts.get("temperature"), top_p=opts.get("top_p"),
              max_tokens=opts.get("num_predict"))
    if req.stream:
        return StreamingResponse(_proxy_stream(req.model, messages, **kw),
                                  media_type="application/x-ndjson")
    result = await _generate_nonstream(req.model, messages, **kw)
    result["response"] = result["message"]["content"]
    return JSONResponse(result)


@app.post("/api/show")
async def show_model(request: Request):
    body = await request.json()
    name = body.get("name", "")
    path = _resolve_model_path(name)
    if not path:
        raise HTTPException(404, f"Model '{name}' not found")
    return {"modelfile": f'FROM {path}', "details": {
        "format": "gguf", "quantization_level": _guess_quant(path.stem)}}


@app.get("/api/ps")
async def running_models():
    if current_model:
        return {"models": [{"name": current_model, "model": current_model,
                            "size": 0, "details": {}}]}
    return {"models": []}


@app.get("/api/version")
async def version():
    return {"version": "0.4.0-turboquant"}


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI-COMPATIBLE PASSTHROUGH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    await _ensure_model(model)
    async with httpx.AsyncClient() as client:
        if body.get("stream", False):
            async def stream_proxy():
                async with client.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                                          json=body, timeout=300) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            return StreamingResponse(stream_proxy(), media_type="text/event-stream")
        r = await client.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                               json=body, timeout=300)
        return JSONResponse(r.json())


@app.get("/v1/models")
async def openai_models():
    models = find_models()
    return {"object": "list",
            "data": [{"id": m["name"], "object": "model", "owned_by": "turboquant"} for m in models]}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/models/list")
async def models_list():
    """Full model list with config info."""
    cfg = _load_config()
    models = find_models()
    for m in models:
        m["is_default"] = m["name"] == cfg.get("default_model")
        m["is_loaded"] = current_model is not None and m["name"].lower() in current_model.lower()
    return {"models": models, "default": cfg.get("default_model"), "current": current_model}


@app.post("/api/models/set-default")
async def set_default_model(request: Request):
    body = await request.json()
    name = body.get("name", "")
    cfg = _load_config()
    cfg["default_model"] = name
    _save_config(cfg)
    return {"status": "ok", "default": name}


@app.post("/api/models/load")
async def load_model(request: Request):
    body = await request.json()
    name = body.get("name", "")
    path = _resolve_model_path(name)
    if not path:
        raise HTTPException(404, f"Model '{name}' not found")
    cfg = _load_config()
    await _start_llama_server(path, cfg.get("gpu_layers", DEFAULT_GPU_LAYERS),
                               cfg.get("ctx_size", DEFAULT_CTX))
    return {"status": "ok", "model": current_model}


@app.post("/api/models/unload")
async def unload_model():
    await _stop_llama_server()
    return {"status": "ok"}


@app.delete("/api/models/delete")
async def delete_model(request: Request):
    body = await request.json()
    name = body.get("name", "")
    path = _resolve_model_path(name)
    if not path or not path.exists():
        raise HTTPException(404, f"Model '{name}' not found")
    if current_model and name.lower() in current_model.lower():
        await _stop_llama_server()
    path.unlink()
    return {"status": "deleted", "name": name}


@app.post("/api/models/update-settings")
async def update_settings(request: Request):
    body = await request.json()
    cfg = _load_config()
    if "gpu_layers" in body:
        cfg["gpu_layers"] = int(body["gpu_layers"])
    if "ctx_size" in body:
        cfg["ctx_size"] = int(body["ctx_size"])
    _save_config(cfg)
    return {"status": "ok", "config": cfg}


# ─── Hardware Info ──────────────────────────────────────────────────────────

@app.get("/api/hardware-info")
async def get_hardware_info():
    """Detect hardware and return compatibility table."""
    vram_total, vram_free = _detect_vram_mb()
    ram_total = psutil.virtual_memory().total / (1024 * 1024)
    gpu_name = _detect_gpu_name()
    table = _generate_compatibility_table(vram_total, ram_total)
    return {
        "gpu_name": gpu_name,
        "vram_total_mb": round(vram_total),
        "vram_free_mb": round(vram_free),
        "ram_total_mb": round(ram_total),
        "compatibility_table": table,
    }


# ─── HuggingFace Search ───────────────────────────────────────────────────────

@app.get("/api/models/search-hf")
async def search_huggingface(q: str = "", limit: int = 20):
    """Search HuggingFace for GGUF models with compatibility info."""
    vram_total, _ = _detect_vram_mb()
    ram_total = psutil.virtual_memory().total / (1024 * 1024)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{HF_API}/models", params={
                "search": q, "filter": "gguf", "sort": "downloads",
                "direction": "-1", "limit": limit,
            })
            if r.status_code != 200:
                return {"results": [], "error": f"HF API returned {r.status_code}"}
            data = r.json()
        results = []
        for m in data:
            mid = m.get("id", "")
            params_b = _guess_params_float(mid)
            compat = calculate_model_fit(params_b, "Q4_K_M", vram_total, ram_total) if params_b > 0 else None
            est_size_gb = round(params_b * 0.56, 1) if params_b > 0 else None
            results.append({
                "id": mid,
                "name": mid.split("/")[-1],
                "author": mid.split("/")[0] if "/" in mid else "",
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "tags": m.get("tags", [])[:5],
                "pipeline_tag": m.get("pipeline_tag", ""),
                "last_modified": m.get("lastModified", ""),
                "params_b": params_b if params_b > 0 else None,
                "estimated_size_gb": est_size_gb,
                "compatibility": compat,
            })
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


def _extract_quant_from_filename(fn: str) -> str:
    """Extract quantization type from a GGUF filename."""
    fn_upper = fn.upper()
    for q in ["IQ2_XXS","IQ2_XS","IQ2_S","IQ2_M","IQ3_XXS","IQ3_XS","IQ3_S",
              "IQ4_XS","IQ4_NL","TQ1_0","TQ2_0",
              "Q2_K_S","Q2_K","Q3_K_S","Q3_K_M","Q3_K_L",
              "Q4_K_S","Q4_K_M","Q4_0","Q4_1",
              "Q5_K_S","Q5_K_M","Q5_0","Q5_1",
              "Q6_K","Q8_0","F16","F32","BF16"]:
        if q in fn_upper:
            return q
    return "Q4_K_M"


@app.get("/api/models/hf-files")
async def hf_model_files(repo: str = ""):
    """List GGUF files in a HuggingFace repo with compatibility info."""
    if not repo:
        raise HTTPException(400, "repo parameter required")
    vram_total, _ = _detect_vram_mb()
    ram_total = psutil.virtual_memory().total / (1024 * 1024)
    params_b = _guess_params_float(repo)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{HF_API}/models/{repo}", params={"blobs": "true"})
            if r.status_code != 200:
                return {"files": [], "error": f"Repo not found: {r.status_code}"}
            data = r.json()
        siblings = data.get("siblings", [])
        files = []
        best_fit_idx = -1
        best_fit_size = 0
        for i, s in enumerate(siblings):
            fn = s.get("rfilename", "")
            if not fn.endswith(".gguf"):
                continue
            quant = _extract_quant_from_filename(fn)
            p = params_b if params_b > 0 else _guess_params_float(fn)
            compat = calculate_model_fit(p, quant, vram_total, ram_total) if p > 0 else None
            fsize = s.get("size", 0)
            files.append({
                "filename": fn,
                "size": fsize,
                "url": f"https://huggingface.co/{repo}/resolve/main/{fn}",
                "quant": quant,
                "compatibility": compat,
            })
            # Track best file that fits in GPU
            if compat and compat["fits"] == "gpu" and fsize > best_fit_size:
                best_fit_size = fsize
                best_fit_idx = len(files) - 1
        # Mark recommended file
        if best_fit_idx >= 0:
            files[best_fit_idx]["recommended"] = True
        elif files:
            # If nothing fits fully in GPU, recommend smallest mixed
            mixed = [(i, f) for i, f in enumerate(files)
                     if f.get("compatibility", {}).get("fits") == "mixed"]
            if mixed:
                mixed.sort(key=lambda x: x[1]["size"])
                files[mixed[0][0]]["recommended"] = True
        return {"files": files, "repo": repo, "params_b": params_b}
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.post("/api/models/download")
async def download_model(request: Request):
    """Start downloading a GGUF file from URL."""
    body = await request.json()
    url = body.get("url", "")
    filename = body.get("filename", "")
    if not url or not filename:
        raise HTTPException(400, "url and filename required")
    if not filename.endswith(".gguf"):
        filename += ".gguf"

    dest = MODELS_DIR / filename
    if dest.exists():
        return {"status": "exists", "path": str(dest)}

    download_id = f"dl_{int(time.time()*1000)}"
    active_downloads[download_id] = {"filename": filename, "url": url, "progress": 0,
                                      "total": 0, "status": "starting", "error": None}

    asyncio.create_task(_download_file(download_id, url, dest))
    return {"status": "started", "download_id": download_id}


async def _download_file(dl_id: str, url: str, dest: Path):
    try:
        active_downloads[dl_id]["status"] = "downloading"
        async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
            async with client.stream("GET", url) as resp:
                total = int(resp.headers.get("content-length", 0))
                active_downloads[dl_id]["total"] = total
                downloaded = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024*1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        active_downloads[dl_id]["progress"] = downloaded
        active_downloads[dl_id]["status"] = "complete"
    except Exception as e:
        active_downloads[dl_id]["status"] = "error"
        active_downloads[dl_id]["error"] = str(e)
        if dest.exists():
            dest.unlink()


@app.get("/api/models/downloads")
async def get_downloads():
    return {"downloads": active_downloads}


@app.get("/api/models/download-status/{dl_id}")
async def download_status(dl_id: str):
    if dl_id not in active_downloads:
        raise HTTPException(404, "Download not found")
    return active_downloads[dl_id]


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION API
# ═══════════════════════════════════════════════════════════════════════════════

QUANT_QUALITY: dict[str, dict] = {
    "Q8_0":    {"bpw": 8.5,  "quality": 99, "speed": 1.0, "label": "Virtually lossless", "tier": "excellent"},
    "Q6_K":    {"bpw": 6.6,  "quality": 97, "speed": 1.2, "label": "Near-perfect quality", "tier": "excellent"},
    "Q5_K_M":  {"bpw": 5.7,  "quality": 95, "speed": 1.3, "label": "Very good quality", "tier": "great"},
    "Q4_K_M":  {"bpw": 4.9,  "quality": 92, "speed": 1.5, "label": "Good balance quality/size", "tier": "great"},
    "Q3_K_M":  {"bpw": 3.9,  "quality": 85, "speed": 1.8, "label": "Acceptable for most tasks", "tier": "good"},
    "Q2_K":    {"bpw": 3.4,  "quality": 72, "speed": 2.0, "label": "Noticeable quality loss, still usable", "tier": "fair"},
    "IQ4_XS":  {"bpw": 4.3,  "quality": 90, "speed": 1.5, "label": "Good quality, imatrix optimized", "tier": "great"},
    "IQ3_XXS": {"bpw": 3.1,  "quality": 78, "speed": 1.9, "label": "Moderate quality loss", "tier": "good"},
    "IQ2_XXS": {"bpw": 2.1,  "quality": 60, "speed": 2.5, "label": "Significant loss, basic tasks only", "tier": "poor"},
    "IQ2_XS":  {"bpw": 2.3,  "quality": 65, "speed": 2.3, "label": "Notable quality loss", "tier": "fair"},
    "IQ1_S":   {"bpw": 1.6,  "quality": 40, "speed": 3.0, "label": "Extreme loss, experimental", "tier": "poor"},
}

# Quantization types safe for requantization (no imatrix needed)
SAFE_REQUANT_TYPES = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K"]

active_quantizations: dict[str, dict] = {}


@app.get("/api/models/quant-types")
async def list_quant_types():
    """List available quantization types with quality info."""
    vram_total, _ = _detect_vram_mb()
    ram_total = psutil.virtual_memory().total / (1024 * 1024)
    types = []
    for qtype, info in QUANT_QUALITY.items():
        safe = qtype in SAFE_REQUANT_TYPES
        types.append({
            "type": qtype, "bpw": info["bpw"], "quality_pct": info["quality"],
            "speed_mult": info["speed"], "label": info["label"], "tier": info["tier"],
            "safe_requant": safe, "needs_imatrix": not safe,
            "bpp": QUANT_BPP.get(qtype, 0.5),
        })
    return {"types": types, "safe_requant": SAFE_REQUANT_TYPES}


@app.post("/api/models/quantize")
async def quantize_model(request: Request):
    """Requantize a local GGUF model to a different quantization type."""
    body = await request.json()
    source_name = body.get("source", "")
    target_type = body.get("target_type", "Q2_K")

    source_path = _resolve_model_path(source_name)
    if not source_path or not source_path.exists():
        raise HTTPException(404, f"Source model '{source_name}' not found")

    # Find quantize binary
    quant_bin = ENGINE_DIR / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        raise HTTPException(500, "llama-quantize binary not found")

    # Validate target type
    if target_type.upper() not in SAFE_REQUANT_TYPES:
        raise HTTPException(400, f"Type '{target_type}' requires an importance matrix. Safe types: {SAFE_REQUANT_TYPES}")

    # Build output filename by replacing old quant suffix with new one
    stem = source_path.stem
    old_quant = _guess_quant(stem)
    if old_quant != "unknown":
        # Replace last occurrence of quant type (case-insensitive)
        pattern = re.compile(re.escape(old_quant), re.IGNORECASE)
        # Find all matches, replace only the last one
        matches = list(pattern.finditer(stem))
        if matches:
            last = matches[-1]
            base_name = stem[:last.start()] + stem[last.end():]
        else:
            base_name = stem
    else:
        base_name = stem
    # Clean up trailing/leading separators
    base_name = base_name.strip(".-_ ")
    # Remove double separators
    base_name = re.sub(r'[.\-_]{2,}', '.', base_name)
    out_name = f"{base_name}.{target_type.upper()}.gguf"
    out_path = MODELS_DIR / out_name

    if out_path.exists():
        return {"status": "exists", "path": str(out_path), "filename": out_name}

    quant_id = f"q_{int(time.time()*1000)}"
    source_size = source_path.stat().st_size
    bpp_source = QUANT_BPP.get(_guess_quant(stem).upper(), 0.56)
    bpp_target = QUANT_BPP.get(target_type.upper(), 0.56)
    est_size = int(source_size * (bpp_target / bpp_source)) if bpp_source > 0 else source_size
    quality = QUANT_QUALITY.get(target_type.upper(), {})

    active_quantizations[quant_id] = {
        "source": source_name, "target_type": target_type.upper(),
        "output": out_name, "status": "starting", "progress": 0,
        "est_size": est_size, "quality_pct": quality.get("quality", 0),
        "quality_label": quality.get("label", ""), "error": None,
    }

    asyncio.create_task(_run_quantization(quant_id, quant_bin, source_path, out_path, target_type.upper()))
    return {"status": "started", "quant_id": quant_id, "output": out_name, "est_size": est_size}


async def _run_quantization(qid: str, quant_bin: Path, source: Path, dest: Path, qtype: str):
    """Run llama-quantize in background."""
    try:
        active_quantizations[qid]["status"] = "quantizing"
        cmd = [str(quant_bin), "--allow-requantize", str(source), str(dest), qtype,
               str(min(os.cpu_count() or 4, 12))]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

        total_tensors = 0
        current_tensor = 0
        async for line in proc.stdout:
            text = line.decode(errors="replace")
            # Parse progress from "[  N/ M]" pattern
            m = re.search(r'\[\s*(\d+)/\s*(\d+)\]', text)
            if m:
                current_tensor = int(m.group(1))
                total_tensors = int(m.group(2))
                active_quantizations[qid]["progress"] = int(current_tensor / total_tensors * 100)
            # Parse final sizes
            if "quant size" in text.lower():
                size_m = re.search(r'([\d.]+)\s*MB', text)
                if size_m:
                    active_quantizations[qid]["final_size"] = int(float(size_m.group(1)) * 1024 * 1024)

        await proc.wait()
        if proc.returncode == 0 and dest.exists():
            active_quantizations[qid]["status"] = "complete"
            active_quantizations[qid]["progress"] = 100
            active_quantizations[qid]["final_size"] = dest.stat().st_size
        else:
            active_quantizations[qid]["status"] = "error"
            active_quantizations[qid]["error"] = f"Process exited with code {proc.returncode}"
            if dest.exists():
                dest.unlink()
    except Exception as e:
        active_quantizations[qid]["status"] = "error"
        active_quantizations[qid]["error"] = str(e)
        if dest.exists():
            dest.unlink()


@app.get("/api/models/quantize-status")
async def quantize_status():
    """Get status of all quantization jobs."""
    return {"quantizations": active_quantizations}


@app.get("/api/models/quantize-status/{qid}")
async def quantize_job_status(qid: str):
    if qid not in active_quantizations:
        raise HTTPException(404, "Quantization job not found")
    return active_quantizations[qid]


# ─── Smart HuggingFace Search ────────────────────────────────────────────────

@app.get("/api/models/smart-search")
async def smart_search_hf(q: str = "", size: str = "auto"):
    """Search HF for models optimized for the user's hardware.
    Automatically suggests the best quantization for the detected hardware.
    """
    vram_total, _ = _detect_vram_mb()
    ram_total = psutil.virtual_memory().total / (1024 * 1024)

    # Determine max model size that fits well
    if size == "auto":
        # For 4GB VRAM: recommend up to 7B Q2_K or 3B Q4_K_M for full GPU
        max_params_gpu = 0
        max_params_mixed = 0
        for pb in [1.7, 3, 7, 8, 13, 27, 70]:
            fit_q4 = calculate_model_fit(pb, "Q4_K_M", vram_total, ram_total)
            fit_q2 = calculate_model_fit(pb, "Q2_K", vram_total, ram_total)
            if fit_q4["fits"] == "gpu":
                max_params_gpu = pb
            if fit_q2["fits"] in ("gpu", "mixed"):
                max_params_mixed = pb
    else:
        max_params_gpu = float(size) if size else 7
        max_params_mixed = max_params_gpu * 3

    # Build search query with GGUF filter
    search_q = q if q else "GGUF"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{HF_API}/models", params={
                "search": search_q, "filter": "gguf", "sort": "downloads",
                "direction": "-1", "limit": 30,
            })
            if r.status_code != 200:
                return {"results": [], "error": f"HF API returned {r.status_code}"}
            data = r.json()

        results = []
        for m in data:
            mid = m.get("id", "")
            params_b = _guess_params_float(mid)

            # Determine best quant for this model
            best_quant = None
            best_fit = None
            if params_b > 0:
                for qt in ["Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XXS"]:
                    fit = calculate_model_fit(params_b, qt, vram_total, ram_total)
                    if fit["fits"] in ("gpu", "mixed"):
                        best_quant = qt
                        best_fit = fit
                        if fit["fits"] == "gpu":
                            break  # Prefer highest quality that fits in GPU

            rec_level = "none"
            if best_fit:
                if best_fit["fits"] == "gpu" and params_b <= max_params_gpu:
                    rec_level = "perfect"
                elif best_fit["fits"] == "gpu":
                    rec_level = "good"
                elif best_fit["fits"] == "mixed":
                    rec_level = "usable"

            results.append({
                "id": mid,
                "name": mid.split("/")[-1],
                "author": mid.split("/")[0] if "/" in mid else "",
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "params_b": params_b if params_b > 0 else None,
                "best_quant": best_quant,
                "best_fit": best_fit,
                "recommendation": rec_level,
                "tags": m.get("tags", [])[:5],
            })

        # Sort: perfect > good > usable > none, then by downloads
        order = {"perfect": 0, "good": 1, "usable": 2, "none": 3}
        results.sort(key=lambda x: (order.get(x["recommendation"], 3), -x["downloads"]))

        return {
            "results": results,
            "hardware": {"vram_mb": round(vram_total), "ram_mb": round(ram_total),
                         "max_gpu_params": max_params_gpu, "max_mixed_params": max_params_mixed},
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK API
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/bench/run")
async def run_benchmark(request: Request):
    """Run inference benchmark on current or specified model."""
    body = await request.json()
    model_name = body.get("model", current_model or "")
    prompt = body.get("prompt", "Explain quantum computing in simple terms.")
    max_tokens = body.get("max_tokens", 128)
    runs = body.get("runs", 3)

    if not model_name:
        raise HTTPException(400, "No model specified or loaded")

    await _ensure_model(model_name)

    results_list = []
    for i in range(runs):
        t0 = time.time()
        async with httpx.AsyncClient() as client:
            r = await client.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                json={"model": model_name,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "stream": False}, timeout=120)
            data = r.json()
        t1 = time.time()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        elapsed = t1 - t0
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        results_list.append({
            "run": i + 1, "elapsed_s": round(elapsed, 3),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "tokens_per_second": round(tps, 1),
            "response_preview": (data.get("choices", [{}])[0]
                                  .get("message", {}).get("content", ""))[:100],
        })

    avg_tps = sum(r["tokens_per_second"] for r in results_list) / len(results_list)
    avg_elapsed = sum(r["elapsed_s"] for r in results_list) / len(results_list)

    # GPU info
    gpu_info = _get_gpu_info()

    result = {
        "model": model_name, "quant": _guess_quant(model_name),
        "params": _guess_params(model_name),
        "prompt": prompt, "max_tokens": max_tokens, "runs": runs,
        "results": results_list,
        "average": {"tokens_per_second": round(avg_tps, 1), "elapsed_s": round(avg_elapsed, 3)},
        "gpu": gpu_info,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    benchmark_results.append(result)

    # Save to file
    bench_file = BENCH_DIR / f"bench_{int(time.time())}.json"
    bench_file.write_text(json.dumps(result, indent=2))

    return result


@app.get("/api/bench/history")
async def bench_history():
    """Get all benchmark results."""
    results = []
    if BENCH_DIR.exists():
        for f in sorted(BENCH_DIR.glob("bench_*.json"), reverse=True):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                continue
    return {"benchmarks": results}


@app.delete("/api/bench/clear")
async def clear_benchmarks():
    if BENCH_DIR.exists():
        for f in BENCH_DIR.glob("bench_*.json"):
            f.unlink()
    benchmark_results.clear()
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS / HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

def _get_gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {"name": parts[0], "vram_total_mb": int(parts[1]), "vram_used_mb": int(parts[2]),
                    "vram_free_mb": int(parts[3]), "utilization": int(parts[4]),
                    "temperature": int(parts[5])}
    except Exception:
        pass
    return {}


@app.get("/health")
async def health():
    return {"status": "ok" if llama_process and llama_process.poll() is None else "no_model",
            "current_model": current_model, "version": "0.4.0-turboquant"}


@app.get("/api/status")
async def status():
    gpu_info = _get_gpu_info()
    mem = psutil.virtual_memory()
    cfg = _load_config()
    return {
        "current_model": current_model, "current_model_path": current_model_path,
        "engine_running": llama_process is not None and llama_process.poll() is None,
        "models_available": len(find_models()),
        "gpu": gpu_info,
        "ram": {"total_mb": mem.total // (1024*1024), "used_mb": mem.used // (1024*1024),
                "available_mb": mem.available // (1024*1024)},
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "config": cfg,
        "active_downloads": len([d for d in active_downloads.values() if d["status"] == "downloading"]),
    }


# ─── Static Files ─────────────────────────────────────────────────────────────

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


def main():
    import uvicorn
    print(f"\n  ⚡ TurboQuant Server v0.4.0")
    print(f"  📡 Ollama API: http://localhost:{API_PORT}")
    print(f"  🌐 Web UI:     http://localhost:{API_PORT}")
    print(f"  📂 Models:     {MODELS_DIR}\n")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    main()
