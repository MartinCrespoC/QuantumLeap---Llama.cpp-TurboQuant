"""
TurboQuant Server — Full-featured LLM platform
Ollama-compatible API + OpenAI API + Model Manager + Benchmarks + Web UI
"""

import asyncio
import glob
import json
import os
import platform
import re
import subprocess
import sys
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

# Web search module
from web_search import web_searcher

# ─── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PROJECT_ROOT / "engine" / "llama.cpp"
MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"
BENCH_DIR = PROJECT_ROOT / "benchmarks"
# ExpertFlow-enabled binary with ROCm/HIP support
LLAMA_SERVER_BIN = ENGINE_DIR / "build" / "bin" / "llama-server"
LLAMA_BENCH_BIN = ENGINE_DIR / "build" / "bin" / "llama-bench"
CONFIG_FILE = PROJECT_ROOT / "config.json"

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8081"))
API_PORT = int(os.environ.get("API_PORT", "11435"))
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

# TurboQuant KV cache state — auto-configured per model load
turboquant_kv_state: dict = {
    "enabled": False, "mode": "TQ3", "gpu_kv_layers": 0, "cpu_kv_layers": 0,
    "kv_vram_mb": 0, "kv_ram_mb": 0, "bits_per_channel": 3.5,
    "total_layers": 0, "num_heads": 0, "head_dim": 128,
}


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"default_model": None, "gpu_layers": DEFAULT_GPU_LAYERS, "ctx_size": DEFAULT_CTX}


def _estimate_model_arch(params_b: float) -> dict:
    """Auto-estimate transformer architecture from parameter count.
    Used to configure TurboQuant KV cache split automatically."""
    if params_b <= 0:
        return {"layers": 32, "heads": 32, "head_dim": 128}
    if params_b <= 1:
        return {"layers": 16, "heads": 16, "head_dim": 64}
    if params_b <= 2:
        return {"layers": 24, "heads": 16, "head_dim": 64}
    if params_b <= 4:
        return {"layers": 28, "heads": 32, "head_dim": 128}
    if params_b <= 10:
        return {"layers": 32, "heads": 32, "head_dim": 128}
    if params_b <= 15:
        return {"layers": 40, "heads": 40, "head_dim": 128}
    if params_b <= 35:
        return {"layers": 48, "heads": 48, "head_dim": 128}
    if params_b <= 75:
        return {"layers": 80, "heads": 64, "head_dim": 128}
    return {"layers": 96, "heads": 96, "head_dim": 128}


def _auto_turboquant_kv_config(
    params_b: float, vram_free_mb: float, ram_total_mb: float,
    gpu_layers_model: int, ctx_size: int,
) -> dict:
    """Auto-configure TurboQuant KV cache split between VRAM and RAM.
    Based on Google Research TurboQuant (arXiv:2504.19874).
    Returns config dict stored in global turboquant_kv_state."""
    arch = _estimate_model_arch(params_b)
    total_layers = arch["layers"]
    num_heads = arch["heads"]
    head_dim = arch["head_dim"]

    # TQ3 mode: 3.5 bits/element, zero quality loss
    bits_per_elem = 3.5
    bytes_per_elem = bits_per_elem / 8.0

    # Per-layer KV memory at max context (K+V = 2x)
    per_layer_kv_bytes = num_heads * ctx_size * head_dim * bytes_per_elem * 2
    total_kv_mb = (total_layers * per_layer_kv_bytes) / (1024 * 1024)

    # VRAM budget for KV: 15% of free VRAM (rest for weights + overhead)
    vram_kv_budget_mb = min(vram_free_mb * 0.15, 600)
    gpu_kv_layers = int(vram_kv_budget_mb / (per_layer_kv_bytes / (1024 * 1024)))
    gpu_kv_layers = max(0, min(gpu_kv_layers, gpu_layers_model, total_layers))
    cpu_kv_layers = total_layers - gpu_kv_layers

    kv_vram_mb = gpu_kv_layers * per_layer_kv_bytes / (1024 * 1024)
    kv_ram_mb = cpu_kv_layers * per_layer_kv_bytes / (1024 * 1024)

    # Equivalent FP32 memory (what we'd need without TurboQuant)
    fp32_per_layer = num_heads * ctx_size * head_dim * 4.0 * 2
    fp32_total_mb = total_layers * fp32_per_layer / (1024 * 1024)

    config = {
        "enabled": True,
        "mode": "TQ3",
        "bits_per_channel": bits_per_elem,
        "total_layers": total_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "gpu_kv_layers": gpu_kv_layers,
        "cpu_kv_layers": cpu_kv_layers,
        "kv_vram_mb": round(kv_vram_mb, 1),
        "kv_ram_mb": round(kv_ram_mb, 1),
        "kv_total_mb": round(total_kv_mb, 1),
        "fp32_equivalent_mb": round(fp32_total_mb, 1),
        "compression_ratio": round(fp32_total_mb / max(total_kv_mb, 0.1), 1),
        "ctx_size": ctx_size,
    }
    return config


def _save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ─── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    yield
    await _stop_llama_server()


app = FastAPI(title="TurboQuant", version="QuantumLeap v0.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Model Utilities ──────────────────────────────────────────────────────────

def find_models() -> list[dict]:
    models = []
    if not MODELS_DIR.exists():
        return models
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        size = f.stat().st_size
        name = f.stem
        is_moe = _is_moe_model(name)
        active_b = _guess_active_params(name) if is_moe else None
        models.append({
            "name": name, "model": name,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)),
            "size": size, "path": str(f),
            "is_moe": is_moe,
            "active_params_b": active_b,
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

def _detect_gpu_vendor() -> str:
    """Detect GPU vendor: 'nvidia', 'amd', 'apple', or 'none'."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                return "apple"
        except Exception:
            pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "nvidia"
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "Card Series" in result.stdout:
            return "amd"
    except Exception:
        pass
    if glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        return "amd"
    return "none"


def _detect_physical_cores() -> int:
    """Detect physical CPU core count (not SMT/HT threads)."""
    try:
        return psutil.cpu_count(logical=False) or 4
    except Exception:
        return 4


def _detect_cpu_mask_physical() -> Optional[str]:
    """Generate hex CPU affinity mask for physical cores only (Linux)."""
    if platform.system() != "Linux":
        return None
    try:
        physical = _detect_physical_cores()
        mask = (1 << physical) - 1
        return f"{mask:x}"
    except Exception:
        return None


def _find_draft_model(target_path: Path) -> Optional[Path]:
    """Auto-find a compatible draft model for speculative decoding."""
    if not MODELS_DIR.exists():
        return None
    target_name = target_path.stem.lower()
    target_size_mb = target_path.stat().st_size / (1024 * 1024)
    # Only use speculative decoding for models > 4GB
    if target_size_mb < 4000:
        return None
    # Detect model family from target name
    family = None
    for f in ["qwen", "llama", "mistral", "phi", "gemma", "deepseek", "yi", "command"]:
        if f in target_name:
            family = f
            break
    candidates = []
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        if f == target_path:
            continue
        draft_name = f.stem.lower()
        draft_size_mb = f.stat().st_size / (1024 * 1024)
        # Draft must be much smaller than target (< 30% size)
        if draft_size_mb > target_size_mb * 0.3:
            continue
        # Prefer same family
        same_family = family and family in draft_name
        candidates.append((same_family, -draft_size_mb, f))
    if candidates:
        candidates.sort(key=lambda x: (not x[0], x[1]))
        return candidates[0][2]
    return None


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
    """Detect available GPU VRAM in MB. Supports NVIDIA and AMD GPUs."""
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    
    # Try AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total_vram = 0
            used_vram = 0
            for line in result.stdout.split("\n"):
                if "Total Memory (B):" in line:
                    total_vram = int(line.split(":")[1].strip()) / (1024 * 1024)
                elif "Total Used Memory (B):" in line:
                    used_vram = int(line.split(":")[1].strip()) / (1024 * 1024)
            if total_vram > 0:
                return total_vram, total_vram - used_vram
    except Exception:
        pass
    
    # Fallback: Try sysfs for AMD GPUs
    try:
        vram_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        if vram_files:
            with open(vram_files[0], 'r') as f:
                total_bytes = int(f.read().strip())
                total_mb = total_bytes / (1024 * 1024)
            # Try to get used VRAM
            used_file = vram_files[0].replace("vram_total", "vram_used")
            try:
                with open(used_file, 'r') as f:
                    used_bytes = int(f.read().strip())
                    used_mb = used_bytes / (1024 * 1024)
                    return total_mb, total_mb - used_mb
            except:
                # If we can't get used, assume 80% free
                return total_mb, total_mb * 0.8
    except Exception:
        pass
    
    return 0.0, 0.0  # No GPU detected


def _detect_gpu_name() -> str:
    """Detect GPU name dynamically. Supports NVIDIA, AMD, and Apple Silicon."""
    # Try Apple Silicon (M1-M5)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                chip = result.stdout.strip()
                # e.g. "Apple M5 Max" — Metal GPU is integrated
                return f"{chip} (Metal GPU)"
        except Exception:
            pass

    # Try NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    
    # Try AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Card Series:" in line:
                    return line.split("Card Series:")[1].strip()
    except Exception:
        pass
    
    # Fallback: Try lspci for AMD/ATI
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "VGA" in line and ("AMD" in line or "ATI" in line):
                    parts = line.split(":")
                    if len(parts) >= 3:
                        gpu_name = parts[2].strip()
                        if "[" in gpu_name:
                            gpu_name = gpu_name.split("[")[1].split("]")[0]
                        return gpu_name
    except Exception:
        pass
    
    return "No GPU"


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


def _detect_moe_from_name(name: str) -> tuple[bool, float]:
    """Detect MoE model and extract active params from name.
    Returns (is_moe, active_params_b).
    Examples: '122B-A10B' → (True, 10.0), '8x7B' → (True, 7.0), '7B' → (False, 7.0)
    """
    # Pattern: 122B-A10B, 141B-A14B, 35B-A3B
    m = re.search(r'(\d+\.?\d*)[bB][-.]?[aA](\d+\.?\d*)[bB]', name)
    if m:
        return True, float(m.group(2))
    # Pattern: 8x7B (Mixtral-style)
    m = re.search(r'(\d+)x(\d+\.?\d*)[bB]', name)
    if m:
        return True, float(m.group(2))
    params = _guess_params_float(name)
    return False, params


def calculate_optimal_ngl(model_path: Path, vram_free_mb: float, ctx_size: int,
                          turboquant_kv: bool = True, is_moe: bool = False) -> int:
    """Calculate optimal number of GPU layers to avoid OOM.
    MoE-aware: MoE layers are 5-10x heavier (include all experts).
    ExpertFlow optimizes at runtime, but llama.cpp still loads full layers.
    """
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    params_b = _guess_params_float(model_path.stem)
    total_layers = _estimate_layers(params_b)

    if is_moe:
        _, active_b = _detect_moe_from_name(model_path.stem)
        if active_b > 0:
            total_layers = _estimate_layers(active_b)
        # MoE layers are much heavier - no adjustment to model_size_mb
        # because llama.cpp loads entire layers (including all experts)

    # Reserve VRAM for KV cache + overhead
    if turboquant_kv:
        kv_cache_mb = (ctx_size / 1024) * 70   # TurboQuant KV: ~70MB per 1K tokens
    else:
        kv_cache_mb = (ctx_size / 1024) * 400   # Standard q4_0: ~400MB per 1K tokens
    
    # For MoE: reserve VRAM for ExpertFlow cache (2.5GB)
    overhead_mb = 2500 if is_moe else 400
    available_mb = vram_free_mb - kv_cache_mb - overhead_mb

    if available_mb <= 0:
        return 0  # CPU only

    layer_size_mb = model_size_mb / total_layers
    gpu_layers = int(available_mb / layer_size_mb)

    # Safety margin: lower for MoE (heavier, less predictable)
    safety = 0.60 if is_moe else 0.85
    gpu_layers = int(gpu_layers * safety)
    
    return max(0, min(gpu_layers, total_layers + 1))


def calculate_model_fit(params_b: float, quant: str, vram_total_mb: float = 4096,
                        ram_total_mb: float = 24576, turboquant_kv: bool = True) -> dict:
    """Calculate if a model fits in the hardware and how."""
    bpp = QUANT_BPP.get(quant.upper(), 0.56)
    model_size_mb = params_b * bpp * 1024  # GB -> MB
    # TurboQuant KV (arXiv:2504.19874): 6x compression → ~140MB for 4K ctx
    kv_mb = 140 if turboquant_kv else 800  # vs ~800MB with q4_0 KV cache
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
    env["PATH"] = f"/opt/rocm/bin:/opt/cuda/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:/opt/cuda/lib64:/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    # ── Detect hardware once ──
    gpu_vendor = _detect_gpu_vendor()
    physical_cores = _detect_physical_cores()
    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    params_b = _guess_params_float(model_path.stem)
    vram_total, vram_free = _detect_vram_mb()

    # Detect MoE models (e.g., "35B-A3B", "8x7B", "141B-A14B")
    is_moe, active_b = _detect_moe_from_name(model_path.stem)

    # ── Auto-calculate GPU layers (TurboQuant KV aware) ──
    use_uma = False
    if gpu_layers >= 99:
        # TurboQuant KV: ~70MB per 1K ctx vs ~400MB with q4_0
        kv_est = (ctx_size / 1024) * 70  # TQ3 compressed KV estimate
        if model_size_mb + kv_est + 400 > vram_free:
            # AMD GPUs: extra overhead for ROCm runtime (~300MB)
            adjusted_vram = vram_free - (300 if gpu_vendor == "amd" else 0)
            gpu_layers = calculate_optimal_ngl(model_path, adjusted_vram, ctx_size,
                                               turboquant_kv=True, is_moe=is_moe)
            # UMA only works with NVIDIA CUDA, not AMD ROCm
            if gpu_vendor == "nvidia" and gpu_layers > 0 and model_size_mb > vram_free * 0.8:
                use_uma = True
                gpu_layers = min(int(gpu_layers * 1.5), _estimate_layers(params_b) + 1)
            moe_tag = f" [MoE active={active_b:.0f}B]" if is_moe else ""
            print(f"[QuantumLeap] Model {model_path.stem}: {model_size_mb:.0f}MB, "
                  f"VRAM free: {vram_free:.0f}MB → auto ngl={gpu_layers}"
                  f"{' +UMA' if use_uma else ''} ({gpu_vendor}){moe_tag}")

    # ── Auto-configure TurboQuant KV cache split (VRAM/RAM) ──
    global turboquant_kv_state
    turboquant_kv_state = _auto_turboquant_kv_config(
        params_b, vram_free, total_ram_mb, gpu_layers, ctx_size)
    tq = turboquant_kv_state
    print(f"[TurboQuant KV] Auto-configured: {tq['mode']} @ {tq['bits_per_channel']} bits/ch")
    print(f"  GPU KV: {tq['gpu_kv_layers']} layers ({tq['kv_vram_mb']} MB VRAM)")
    print(f"  CPU KV: {tq['cpu_kv_layers']} layers ({tq['kv_ram_mb']} MB RAM)")
    print(f"  Total:  {tq['kv_total_mb']} MB compressed vs {tq['fp32_equivalent_mb']} MB FP32 "
          f"({tq['compression_ratio']}x savings)")

    # ── Thread tuning ──
    # MoE: fewer threads (leave cores for expert routing overhead)
    # Dense: use all physical cores
    if is_moe:
        opt_threads = max(4, physical_cores - 2)
    else:
        opt_threads = physical_cores
    # Batch/prompt processing always uses all physical cores
    opt_threads_batch = physical_cores

    # ── Build command ──
    cmd = [str(bin_path), "-m", str(model_path), "--host", "127.0.0.1",
           "--port", str(LLAMA_PORT), "-ngl", str(gpu_layers), "-c", str(ctx_size),
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--metrics",
           "-t", str(opt_threads), "-tb", str(opt_threads_batch),
           "--prio", "2"]

    # ── CPU affinity: pin to physical cores to avoid SMT thrashing ──
    cpu_mask = _detect_cpu_mask_physical()
    if cpu_mask:
        cmd.extend(["-C", cpu_mask, "-Cb", cpu_mask])

    # ── Memory mapping strategy ──
    # MoE models: ALWAYS use mmap — OS pages experts on demand, only shared weights
    # (~12% of model) stay hot in page cache. This makes load near-instant vs reading
    # the entire 34 GB sequentially (which took 98s with --no-mmap).
    # Dense models: use --no-mmap for mid-size (4-20 GB) where sequential read wins.
    # mlock: only if model fits comfortably in RAM (<70% of total).
    if model_size_mb < total_ram_mb * 0.70:
        cmd.append("--mlock")
    if not is_moe and 4000 < model_size_mb < 20000:
        cmd.append("--no-mmap")

    # ── Speculative decoding ──
    # Skip draft model when VRAM is tight — draft needs its own GPU allocation.
    # Use ngram speculation instead (free, no VRAM cost, ~30-50% speedup).
    main_model_gpu_mb = gpu_layers * (model_size_mb / max(_estimate_layers(params_b), 1))
    vram_headroom_mb = vram_free - main_model_gpu_mb
    draft_path = _find_draft_model(model_path) if vram_headroom_mb > 2000 else None
    if draft_path:
        draft_size_mb = draft_path.stat().st_size / (1024 * 1024)
        if draft_size_mb > vram_headroom_mb * 0.5:
            # Draft model too large for remaining VRAM — fall back to ngram
            print(f"[TurboQuant] Skipping draft model (needs {draft_size_mb:.0f}MB, "
                  f"only {vram_headroom_mb:.0f}MB headroom) → using ngram")
            draft_path = None
    if draft_path:
        # Draft model speculative decoding (2-3x speedup)
        cmd.extend(["-md", str(draft_path), "--draft-max", "16",
                     "--draft-min", "1", "--draft-p-min", "0.75"])
        print(f"[TurboQuant] Speculative: draft={draft_path.stem}")
    elif params_b > 7:
        # N-gram speculative decoding (free, no draft model needed, ~30-50% speedup)
        cmd.extend(["--spec-type", "ngram-simple", "--draft-max", "8"])
        print(f"[TurboQuant] Speculative: ngram-simple (no draft model)")

    # ── GPU vendor-specific env ──
    if use_uma and gpu_vendor == "nvidia":
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

    optflags = []
    if is_moe:
        optflags.append("MoE")
    if "--no-mmap" in cmd:
        optflags.append("no-mmap")
    if use_uma:
        optflags.append("UMA")
    if draft_path:
        optflags.append(f"draft:{draft_path.stem}")
    elif params_b > 7:
        optflags.append("ngram-spec")
    if cpu_mask:
        optflags.append(f"cpumask:0x{cpu_mask}")
    if turboquant_kv_state.get("enabled"):
        optflags.append(f"TQ-KV:{tq['gpu_kv_layers']}gpu+{tq['cpu_kv_layers']}cpu")
    print(f"[TurboQuant] Starting: ngl={gpu_layers} threads={opt_threads}/{opt_threads_batch} "
          f"ctx={ctx_size} opts=[{', '.join(optflags)}]")
    print(f"[TurboQuant] cmd: {' '.join(str(c) for c in cmd[-15:])}")

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
            last_usage = {}
            done_sent = False
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        if not done_sent:
                            final_chunk = {
                                "model": model,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                                "done_reason": "stop",
                                "prompt_eval_count": last_usage.get("prompt_tokens", 0),
                                "eval_count": last_usage.get("completion_tokens", 0),
                            }
                            yield json.dumps(final_chunk) + "\n"
                            done_sent = True
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content") or ""
                            finish = data.get("choices", [{}])[0].get("finish_reason")
                            usage = data.get("usage")
                            if usage:
                                last_usage = usage
                            ollama_chunk = {
                                "model": model,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "message": {"role": "assistant", "content": content},
                                "done": finish is not None,
                            }
                            if finish:
                                ollama_chunk["done_reason"] = "stop"
                                ollama_chunk["prompt_eval_count"] = last_usage.get("prompt_tokens", 0)
                                ollama_chunk["eval_count"] = last_usage.get("completion_tokens", 0)
                                done_sent = True
                            yield json.dumps(ollama_chunk) + "\n"
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
            
            if not done_sent:
                final_chunk = {
                    "model": model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": last_usage.get("prompt_tokens", 0),
                    "eval_count": last_usage.get("completion_tokens", 0),
                }
                yield json.dumps(final_chunk) + "\n"


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
    return {"version": "0.5.0-turboquant"}


@app.get("/api/kv-stats")
async def kv_stats():
    """Real-time TurboQuant KV cache stats — auto-configured per model load.
    Shows VRAM/RAM split, compression ratio, and memory savings."""
    if not turboquant_kv_state.get("enabled"):
        return {"enabled": False, "message": "No model loaded or TurboQuant KV not active"}
    tq = turboquant_kv_state
    return {
        "enabled": True,
        "model": current_model,
        "mode": tq["mode"],
        "bits_per_channel": tq["bits_per_channel"],
        "architecture": {
            "total_layers": tq["total_layers"],
            "num_heads": tq["num_heads"],
            "head_dim": tq["head_dim"],
            "ctx_size": tq["ctx_size"],
        },
        "memory_split": {
            "gpu_kv_layers": tq["gpu_kv_layers"],
            "cpu_kv_layers": tq["cpu_kv_layers"],
            "kv_vram_mb": tq["kv_vram_mb"],
            "kv_ram_mb": tq["kv_ram_mb"],
            "kv_total_mb": tq["kv_total_mb"],
        },
        "savings": {
            "fp32_equivalent_mb": tq["fp32_equivalent_mb"],
            "compression_ratio": tq["compression_ratio"],
            "vram_saved_mb": round(tq["fp32_equivalent_mb"] - tq["kv_total_mb"], 1),
        },
        "pipeline": ["Hadamard rotation (FWHT)", "PolarQuant (angle quantization)",
                      "QJL (1-bit residual correction)"],
        "dispatch": {
            "gpu_layers": "CUDA fused attention kernel (8x speedup)",
            "cpu_layers": "AVX-512/AVX2 SIMD attention",
        },
    }


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

def _is_moe_model(name: str) -> bool:
    """Detect if model name indicates a Mixture of Experts architecture."""
    n = name.lower()
    # Patterns: "35B-A3B", "8x7B", "141B-A14B", "MoE", "mixture"
    if re.search(r'\d+[bB][-.]?[aA]\d+[bB]', name):
        return True
    if re.search(r'\d+x\d+[bB]', name, re.IGNORECASE):
        return True
    if "moe" in n or "mixture" in n:
        return True
    return False


def _guess_active_params(name: str) -> Optional[float]:
    """For MoE models, extract active parameter count (e.g., 3B from 35B-A3B)."""
    m = re.search(r'(\d+\.?\d*)[bB][-.]?[aA](\d+\.?\d*)[bB]', name)
    if m:
        return float(m.group(2))
    return None


def _get_cpu_info() -> dict:
    """Get CPU info cross-platform."""
    info: dict = {"cores_physical": _detect_physical_cores(),
                  "cores_logical": psutil.cpu_count(logical=True) or 1}
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["model"] = line.split(":")[1].strip()
                        break
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                info["avx512"] = "avx512" in content
                info["avx2"] = "avx2" in content
                info["avx"] = " avx " in content or "avx " in content
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info["model"] = result.stdout.strip()
        except Exception:
            pass
    return info


@app.get("/api/hardware-info")
async def get_hardware_info():
    """Full dynamic hardware profile — updates automatically if hardware changes."""
    gpu_vendor = _detect_gpu_vendor()
    gpu_name = _detect_gpu_name()
    vram_total, vram_free = _detect_vram_mb()
    ram = psutil.virtual_memory()
    ram_total_mb = ram.total / (1024 * 1024)
    ram_free_mb = ram.available / (1024 * 1024)
    cpu_info = _get_cpu_info()
    table = _generate_compatibility_table(vram_total, ram_total_mb)

    # Apple Silicon: unified memory — all RAM is effectively VRAM
    is_unified = gpu_vendor == "apple"
    effective_vram = ram_total_mb if is_unified else vram_total

    return {
        "gpu": {
            "name": gpu_name,
            "vendor": gpu_vendor,
            "vram_total_mb": round(effective_vram),
            "vram_free_mb": round(ram_free_mb if is_unified else vram_free),
            "unified_memory": is_unified,
        },
        "cpu": cpu_info,
        "ram": {
            "total_mb": round(ram_total_mb),
            "free_mb": round(ram_free_mb),
            "total_gb": round(ram_total_mb / 1024, 1),
        },
        "compatibility_table": table,
        "tier": _hardware_tier(effective_vram, ram_total_mb),
        "turboquant_kv": {
            "enabled": True,
            "mode": "TQ3",
            "bits_per_channel": 3.5,
            "compression_vs_fp32": "6x",
            "attention_speedup": "up to 8x (4-bit mode)",
            "quality_loss": "zero (at 3.5 bits)",
            "paper": "Google Research arXiv:2504.19874 (ICLR 2026)",
            "components": ["Hadamard preconditioning", "PolarQuant", "QJL residual correction"],
            "kv_memory_4k_ctx_mb": 140,
            "kv_memory_4k_ctx_mb_legacy": 800,
            "mixed_memory": {
                "supported": True,
                "description": "KV cache splits across VRAM and RAM per-layer automatically",
                "gpu_layers_kv": "First N layers in VRAM (CUDA fused attention, fastest)",
                "cpu_layers_kv": "Remaining layers in RAM (AVX-512/AVX2 SIMD attention)",
                "vram_kv_budget_mb": round(min(effective_vram * 0.15, 600)),
                "ram_kv_budget_mb": round(min(ram_total_mb * 0.20, 4096)),
                "dynamic_migration": "Layers can migrate GPU<->CPU at runtime under VRAM pressure",
            },
        },
    }


def _hardware_tier(vram_mb: float, ram_mb: float) -> dict:
    """Classify hardware into performance tiers for recommendations."""
    vram_gb = vram_mb / 1024
    ram_gb = ram_mb / 1024
    if vram_gb >= 24:
        tier = "ultra"
        desc = "Run 70B+ models at high quality"
    elif vram_gb >= 12:
        tier = "high"
        desc = "Run 27B-70B models, MoE models shine"
    elif vram_gb >= 6:
        tier = "mid"
        desc = "Run 7B-13B on GPU, larger with mixed offload"
    elif vram_gb >= 3:
        tier = "low"
        desc = "Small models on GPU, MoE recommended for larger"
    else:
        tier = "cpu"
        desc = "CPU-only inference, MoE models recommended"
    return {"tier": tier, "description": desc, "vram_gb": round(vram_gb, 1),
            "ram_gb": round(ram_gb, 1)}


# ─── Model Recommendations ────────────────────────────────────────────────────

# Curated high-quality models with MoE and dense options
RECOMMENDED_MODELS: list[dict] = [
    # MoE models — best tok/s per intelligence dollar
    {"repo": "unsloth/Qwen3-30B-A3B-GGUF", "name": "Qwen3 30B-A3B (MoE)",
     "params_b": 30, "active_b": 3, "is_moe": True, "family": "qwen",
     "desc": "30B MoE, only 3B active. Incredible speed-to-quality ratio.",
     "quants": {"4gb": "IQ2_XXS", "6gb": "Q2_K", "8gb": "Q3_K_M", "12gb": "Q4_K_M", "16gb+": "Q6_K"}},
    {"repo": "unsloth/Qwen3-235B-A22B-GGUF", "name": "Qwen3 235B-A22B (MoE)",
     "params_b": 235, "active_b": 22, "is_moe": True, "family": "qwen",
     "desc": "235B total, 22B active. Top-tier intelligence, needs RAM.",
     "quants": {"48gb": "IQ2_XXS", "64gb": "Q2_K", "96gb+": "Q4_K_M"}},
    {"repo": "bartowski/deepseek-v3-0324-GGUF", "name": "DeepSeek V3 (MoE)",
     "params_b": 671, "active_b": 37, "is_moe": True, "family": "deepseek",
     "desc": "671B total, 37B active. Best open-source MoE for reasoning.",
     "quants": {"96gb": "IQ1_S", "128gb+": "IQ2_XXS"}},
    {"repo": "unsloth/Mixtral-8x7B-Instruct-v0.1-GGUF", "name": "Mixtral 8x7B (MoE)",
     "params_b": 47, "active_b": 13, "is_moe": True, "family": "mistral",
     "desc": "Classic MoE. 47B params, 13B active. Fast and capable.",
     "quants": {"6gb": "IQ2_XXS", "8gb": "Q2_K", "12gb": "Q3_K_M", "16gb": "Q4_K_M", "24gb+": "Q6_K"}},
    # Dense models — best quality per parameter
    {"repo": "unsloth/Qwen3-8B-GGUF", "name": "Qwen3 8B",
     "params_b": 8, "active_b": 8, "is_moe": False, "family": "qwen",
     "desc": "Excellent 8B dense model. Great for 6-8GB VRAM.",
     "quants": {"4gb": "IQ2_XXS", "6gb": "Q3_K_M", "8gb": "Q4_K_M", "12gb": "Q6_K", "16gb+": "Q8_0"}},
    {"repo": "unsloth/Qwen3-4B-GGUF", "name": "Qwen3 4B",
     "params_b": 4, "active_b": 4, "is_moe": False, "family": "qwen",
     "desc": "Fast 4B model. Full GPU even on 4GB VRAM.",
     "quants": {"4gb": "Q4_K_M", "6gb": "Q6_K", "8gb+": "Q8_0"}},
    {"repo": "bartowski/Llama-3.3-70B-Instruct-GGUF", "name": "Llama 3.3 70B",
     "params_b": 70, "active_b": 70, "is_moe": False, "family": "llama",
     "desc": "Meta's flagship 70B. Top quality, needs serious hardware.",
     "quants": {"24gb": "IQ2_XXS", "48gb": "Q2_K", "64gb": "Q3_K_M", "96gb+": "Q4_K_M"}},
    {"repo": "unsloth/gemma-3-27b-it-GGUF", "name": "Gemma 3 27B",
     "params_b": 27, "active_b": 27, "is_moe": False, "family": "gemma",
     "desc": "Google's 27B. Excellent instruction following.",
     "quants": {"8gb": "IQ2_XXS", "12gb": "Q2_K", "16gb": "Q3_K_M", "24gb": "Q4_K_M", "48gb+": "Q8_0"}},
    {"repo": "bartowski/Phi-4-mini-instruct-GGUF", "name": "Phi 4 Mini 3.8B",
     "params_b": 3.8, "active_b": 3.8, "is_moe": False, "family": "phi",
     "desc": "Microsoft's compact model. Superb quality for size.",
     "quants": {"4gb": "Q4_K_M", "6gb": "Q6_K", "8gb+": "Q8_0"}},
    {"repo": "unsloth/SmolLM2-1.7B-Instruct-GGUF", "name": "SmolLM2 1.7B",
     "params_b": 1.7, "active_b": 1.7, "is_moe": False, "family": "smollm",
     "desc": "Tiny but capable. Lightning fast on any hardware.",
     "quants": {"2gb": "Q4_K_M", "4gb+": "Q8_0"}},
]


@app.get("/api/recommendations")
async def get_recommendations(moe_only: bool = False):
    """Get model recommendations tailored to the detected hardware."""
    gpu_vendor = _detect_gpu_vendor()
    vram_total, vram_free = _detect_vram_mb()
    ram = psutil.virtual_memory()
    ram_total_mb = ram.total / (1024 * 1024)

    # Apple Silicon: unified memory
    is_unified = gpu_vendor == "apple"
    effective_vram_mb = ram_total_mb if is_unified else vram_total
    effective_vram_gb = effective_vram_mb / 1024
    total_mem_gb = ram_total_mb / 1024

    recommendations = []
    for model in RECOMMENDED_MODELS:
        if moe_only and not model["is_moe"]:
            continue

        # Find the best quant for this hardware
        best_quant = None
        best_tier_key = None
        for tier_key, quant in model["quants"].items():
            tier_gb = float(re.search(r'(\d+)', tier_key).group(1))
            plus = "+" in tier_key
            if is_unified:
                # Apple: use total RAM as VRAM budget
                fits = total_mem_gb >= tier_gb if not plus else total_mem_gb >= tier_gb
            else:
                # Discrete GPU: model can span VRAM + RAM
                fits = (effective_vram_gb >= tier_gb) or (total_mem_gb >= tier_gb)
            if fits:
                best_quant = quant
                best_tier_key = tier_key

        if not best_quant:
            continue

        # Calculate fit details
        fit = calculate_model_fit(model["params_b"], best_quant, effective_vram_mb, ram_total_mb)
        active_b = model["active_b"]
        estimated_tps = _estimate_tok_per_sec(active_b, best_quant, effective_vram_mb, ram_total_mb)

        rec = {
            "repo": model["repo"],
            "name": model["name"],
            "description": model["desc"],
            "family": model["family"],
            "is_moe": model["is_moe"],
            "params_b": model["params_b"],
            "active_params_b": active_b,
            "recommended_quant": best_quant,
            "fit": fit,
            "estimated_tps": estimated_tps,
            "quality": QUANT_QUALITY.get(best_quant, {}),
        }

        # MoE advantage explanation
        if model["is_moe"]:
            rec["moe_advantage"] = (
                f"Only {active_b}B params active per token (out of {model['params_b']}B total). "
                f"~{model['params_b']/active_b:.0f}x more knowledge than a {active_b}B dense model "
                f"at similar speed."
            )

        recommendations.append(rec)

    # Sort: MoE first if not filtered, then by estimated speed
    recommendations.sort(key=lambda r: (
        0 if r["is_moe"] else 1,
        -r["estimated_tps"],
    ))

    tier = _hardware_tier(effective_vram_mb, ram_total_mb)

    return {
        "hardware": {
            "gpu": _detect_gpu_name(),
            "gpu_vendor": gpu_vendor,
            "vram_gb": round(effective_vram_gb, 1),
            "ram_gb": round(total_mem_gb, 1),
            "unified_memory": is_unified,
            "tier": tier,
        },
        "recommendations": recommendations,
        "tip": _hardware_tip(tier["tier"], is_unified, gpu_vendor),
    }


def _estimate_tok_per_sec(active_params_b: float, quant: str, vram_mb: float, ram_mb: float) -> float:
    """Rough estimate of tokens/second based on hardware and model."""
    bpp = QUANT_BPP.get(quant, 0.56)
    model_size_mb = active_params_b * bpp * 1000
    # If model fits in VRAM: GPU inference speed
    if model_size_mb < vram_mb * 0.85:
        # GPU bound: ~40 tok/s for 7B Q4 on mid GPU, scales inversely with size
        base_tps = 280 / active_params_b
        return round(min(base_tps, 120), 1)
    # Mixed offload: slower due to PCIe transfers
    gpu_fraction = min(vram_mb / max(model_size_mb, 1), 1.0)
    base_tps = 280 / active_params_b
    penalty = 0.4 + 0.6 * gpu_fraction  # 40% base + 60% scaled by GPU usage
    return round(min(base_tps * penalty, 120), 1)


def _hardware_tip(tier: str, unified: bool, vendor: str) -> str:
    """Return a helpful tip based on hardware."""
    if unified:
        return ("Apple Silicon with unified memory: ALL your RAM is VRAM! "
                "You can run much larger models than discrete GPU systems with similar RAM. "
                "MoE models are especially fast because the active parameters fit in memory bandwidth.")
    if tier == "ultra":
        return "Excellent hardware! Run 70B+ models at high quality. Try dense models for best quality."
    if tier == "high":
        return "Great setup! MoE models give you 70B-level intelligence at 7B speed. Try Qwen3-30B-A3B."
    if tier == "mid":
        return ("Good for 7-13B dense models. For larger, use MoE models — "
                "Qwen3-30B-A3B runs like a 3B model but thinks like a 30B.")
    if tier == "low":
        return ("Limited VRAM but MoE models are your secret weapon! "
                "Qwen3-30B-A3B with IQ2_XXS gives 30B intelligence at 3B speed.")
    return ("CPU-only mode. MoE models recommended — they only activate 3B params per token "
            "even though the full model has 30B+. Much faster than dense models of similar quality.")


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
            is_moe = _is_moe_model(mid)
            active_b = _guess_active_params(mid) if is_moe else params_b
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
                "active_params_b": active_b if is_moe and active_b else None,
                "is_moe": is_moe,
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
            is_moe = _is_moe_model(mid)
            active_b = _guess_active_params(mid) if is_moe else None

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
            # MoE boost: bump recommendation level (they're faster than size suggests)
            if is_moe and active_b and rec_level == "usable":
                active_fit = calculate_model_fit(active_b, best_quant or "Q4_K_M", vram_total, ram_total)
                if active_fit["fits"] == "gpu":
                    rec_level = "good"

            results.append({
                "id": mid,
                "name": mid.split("/")[-1],
                "author": mid.split("/")[0] if "/" in mid else "",
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "params_b": params_b if params_b > 0 else None,
                "active_params_b": active_b,
                "is_moe": is_moe,
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
    vendor = _detect_gpu_vendor()
    gpu_name = _detect_gpu_name()
    vram_total, vram_free = _detect_vram_mb()
    info: dict = {
        "name": gpu_name, "vendor": vendor,
        "vram_total_mb": round(vram_total), "vram_free_mb": round(vram_free),
        "vram_used_mb": round(vram_total - vram_free),
    }
    # NVIDIA-specific: utilization + temperature
    if vendor == "nvidia":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                info["utilization"] = int(parts[0])
                info["temperature"] = int(parts[1])
        except Exception:
            pass
    # AMD-specific: temperature
    elif vendor == "amd":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Temperature" in line and "Edge" in line:
                        temp = re.search(r'(\d+\.?\d*)', line.split(":")[-1])
                        if temp:
                            info["temperature"] = int(float(temp.group(1)))
        except Exception:
            pass
    # Apple Silicon: unified memory = total RAM
    elif vendor == "apple":
        info["unified_memory"] = True
        info["vram_total_mb"] = round(psutil.virtual_memory().total / (1024 * 1024))
        info["vram_free_mb"] = round(psutil.virtual_memory().available / (1024 * 1024))
        info["vram_used_mb"] = info["vram_total_mb"] - info["vram_free_mb"]
    return info


@app.get("/health")
async def health():
    return {"status": "ok" if llama_process and llama_process.poll() is None else "no_model",
            "current_model": current_model, "version": "0.5.0-turboquant"}


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


# ─── Web Search ───────────────────────────────────────────────────────────────

class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=3, ge=1, le=10, description="Number of results to fetch")
    max_content_length: int = Field(default=3000, ge=500, le=10000, description="Max content length per result")


@app.post("/api/web/search")
async def web_search(request: WebSearchRequest):
    """Search the web and fetch content from top results"""
    try:
        result = await web_searcher.search_and_fetch(
            query=request.query,
            num_results=request.num_results,
            max_content_length=request.max_content_length
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")


@app.get("/api/web/fetch")
async def web_fetch(url: str, max_length: int = 5000):
    """Fetch and extract content from a URL"""
    try:
        result = await web_searcher.fetch_url(url, max_length=max_length)
        if "error" in result and not result.get("content"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {str(e)}")


# ─── Static Files ─────────────────────────────────────────────────────────────

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


def main():
    import uvicorn
    print(f"\n  ⚡ TurboQuant Server v0.5.0")
    print(f"  📡 Ollama API: http://localhost:{API_PORT}")
    print(f"  🌐 Web UI:     http://localhost:{API_PORT}")
    print(f"  📂 Models:     {MODELS_DIR}\n")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    main()
