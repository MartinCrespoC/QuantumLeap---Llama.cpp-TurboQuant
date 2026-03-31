# Resumen de Cambios — QuantumLeap v0.5.0

**Fecha**: 30 Marzo 2026

---

## v0.5.0 — Ollama Coexistence + TurboQuant KV Engine

### Coexistencia con Ollama

- **Puerto cambiado**: 11434 → **11435** (Ollama se queda en 11434, sin conflictos)
- **Scripts reescritos**: `start.sh`, `start.bat`, `start_mac.sh` solo matan procesos de QuantumLeap en su puerto
- **Kill scripts**: `kill-server.sh/bat/mac.sh` solo detienen QuantumLeap + llama-server interno (8081)
- **Variable `API_PORT`**: Configurable. `API_PORT=11434 bash scripts/start.sh` para reemplazar Ollama
- **`api/server.py`**: Puerto por defecto cambiado a 11435

### TurboQuant KV Cache — Motor C++/CUDA

Implementacion completa del paper [TurboQuant](https://arxiv.org/abs/2504.19874) (Google, ICLR 2026):

**Pipeline**: Hadamard FWHT → PolarQuant → QJL 1-bit residual

**CPU (AVX2)**:
- QJL encode: FMA dot product con doble acumulador, 64-bit word packing
- QJL inner product: Branchless accumulation (`2*pos - total`)
- PolarQuant: Vectorized squared norm + residual subtraction
- Hadamard: Vectorized sign-flip + normalization (forward + inverse)
- Prefetch hints para acceso secuencial KV

**CUDA (SM 8.6)**:
- Attention kernel: query + projections en shared memory, LUT dequantizacion pre-computado
- Fused `__sincosf` para trigonometria
- Warp shuffle reductions
- `fused_residual_qjl_kernel`: polar reconstruct + residual + QJL sign extraction en un solo kernel (elimina CPU fallback)

**Memory**:
- `TQCompressedKV::reserve()` — pre-aloca buffers para max seq length
- `turboquant_kv_append()` — stack buffers para single-vector (zero heap alloc en generacion autoregresiva)

**Resultados**:
- 16/16 tests passing
- 14-25x speedup AVX2 en polar transform
- TQ3: 7.4x compresion, 3.5 bits/channel, zero quality loss
- TQ2: 9.7x compresion, 2.5 bits/channel

### Bugfix

- **`residual_quantize`**: Escalas se acumulaban entre iteraciones en vez de sobreescribirse → MSE inflado. Corregido.

### Documentacion

- **README.md**: Reescrito limpio, profesional, con seccion Ollama coexistence
- **QUICKSTART.md**: Simplificado, 5 min setup, sin conflictos
- **SETUP_COMPLETO.md**: Guia completa, Ollama-friendly
- **Eliminados** emojis excesivos y texto redundante

### Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `api/server.py` | Puerto 11434 → 11435 |
| `scripts/start.sh` | Puerto 11435, no mata Ollama |
| `scripts/start.bat` | Puerto 11435, no mata Ollama |
| `scripts/start_mac.sh` | Puerto 11435, no mata Ollama |
| `scripts/kill-server.sh` | Solo mata QuantumLeap + port 8081 |
| `scripts/kill-server.bat` | Solo mata QuantumLeap + port 8081 |
| `scripts/kill-server-mac.sh` | Solo mata QuantumLeap + port 8081 |
| `core/src/turboquant/turboquant_kv.cpp` | `reserve()` + optimized `append()` |
| `core/src/turboquant/residual_quant.cpp` | Bugfix acumulacion de escalas |
| `core/src/turboquant/turboquant_kv_cuda.cu` | Shared memory, fused kernels |
| `core/src/turboquant/qjl.cpp` | AVX2 + 64-bit word packing |
| `core/src/turboquant/polarquant.cpp` | AVX2 norm + residual |
| `core/src/turboquant/hadamard.cpp` | AVX2 sign-flip + normalize |

---

## v0.4.0 — Rebrand + Optimizacion

- Rename TurboQuant → QuantumLeap (TurboQuant queda como motor interno)
- Scripts idempotentes (setup multiples veces sin error)
- Deteccion de hardware 100% dinamica
- Auto-kill de procesos en puerto al iniciar
- Cross-platform: Linux, Windows, macOS
