# QuantumLeap — Rebrand & Version History

## Naming

- **QuantumLeap** — Project name (user-facing)
- **TurboQuant** — Internal optimization engine name (code-facing)
- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) via [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork

## v0.5.0 (March 30, 2026)

**Ollama coexistence** — QuantumLeap now runs on port 11435 by default, alongside Ollama on 11434. No conflicts, no need to uninstall anything.

**TurboQuant KV Cache Engine** — Full C++/CUDA implementation of Google's TurboQuant paper (arXiv:2504.19874):
- Hadamard FWHT + PolarQuant + QJL pipeline
- AVX2 SIMD vectorization (14-25x speedup)
- CUDA shared memory attention, fused kernels, warp shuffles
- Zero-alloc autoregressive append with pre-reserved buffers
- 16/16 tests passing, TQ3 at 7.4x compression

**Scripts rewritten** — All start/stop scripts only manage QuantumLeap processes. Never touch Ollama.

**Docs cleaned up** — README, QUICKSTART, SETUP rewritten. Removed emoji clutter and redundant files.

## v0.4.0 (March 30, 2026)

**Rebrand** — TurboQuant project → QuantumLeap. TurboQuant remains as engine name.

**Auto-optimization engine**:
- UMA cliff detection (+107%)
- --no-mmap for MoE (+28%)
- Thread tuning (8 dense / 6 MoE)
- mlock, q4_0 KV cache
- MoE auto-detection

**Benchmarks** (RTX 3050 4GB + i5-11400H + 24GB DDR4):
- MoE 35B-A3B IQ2_XXS: 15.68 tok/s
- Dense 27B Q2_K: 4.20 tok/s
- 4B Q2_K: 44.80 tok/s

## Key Learnings

1. MoE > Dense for constrained hardware (3x faster)
2. --no-mmap critical for MoE (eliminates page faults)
3. Memory bandwidth is the bottleneck, not CPU/GPU compute
4. UMA cliff detection essential (wrong ngl = 50% loss)
5. Speculative decoding doesn't work on 4GB VRAM
