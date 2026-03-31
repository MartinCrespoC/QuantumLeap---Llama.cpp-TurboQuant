# QuantumLeap — Setup Completo

**Version**: 0.5.0  
**Coexiste con Ollama** — no es necesario desinstalar nada.

---

## Requisitos

| Componente | Minimo | Recomendado |
|------------|--------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 8GB | 16GB+ |
| GPU | Opcional | NVIDIA (CUDA) / AMD (ROCm) / Apple Metal |
| OS | Linux, Windows 10+, macOS | — |

## Instalacion

```bash
git clone https://github.com/MartinCrespoC/QuantumLeap.git
cd QuantumLeap
bash scripts/setup.sh
```

El script `setup.sh` hace todo automaticamente:
1. Detecta OS, CPU, GPU, SIMD (AVX2/AVX-512)
2. Instala cmake y ninja si faltan
3. Detecta CUDA / ROCm / Metal y configura el build
4. Compila llama.cpp con flags optimos
5. Crea entorno virtual Python e instala dependencias
6. Descarga un modelo de prueba (SmolLM2 1.7B, ~1GB)

**Tiempo**: 5-10 minutos (la mayoria compilando llama.cpp).

## Iniciar Servidor

```bash
# Linux
bash scripts/start.sh

# Windows
scripts\start.bat

# macOS
bash scripts/start_mac.sh
```

Web UI en **http://localhost:11435**.

## Coexistencia con Ollama

QuantumLeap usa puerto **11435** por defecto. Ollama usa **11434**. Ambos funcionan simultaneamente sin conflictos.

```bash
# Los dos corriendo al mismo tiempo — sin problemas
ollama serve                     # :11434
bash scripts/start.sh            # :11435
```

**Reemplazar Ollama** (mismo puerto):
```bash
API_PORT=11434 bash scripts/start.sh
```

Los scripts de stop (`kill-server.sh`) solo detienen QuantumLeap. **Nunca tocan Ollama**.

## Deteccion de Hardware

El setup detecta automaticamente:

| Componente | Deteccion |
|------------|-----------|
| GPU NVIDIA | `nvidia-smi` → nombre, VRAM, Compute Capability |
| GPU AMD | `rocm-smi` o `lspci` → nombre, VRAM via sysfs |
| Apple Metal | `system_profiler` → chip, GPU cores, memoria unificada |
| CPU | `/proc/cpuinfo` o `sysctl` → modelo, cores |
| SIMD | flags: AVX, AVX2, AVX-512 |
| RAM | `free` o `sysctl hw.memsize` |

## Auto-Optimizacion (TurboQuant Engine)

Al cargar un modelo, se aplican optimizaciones automaticamente:

- **UMA** — Mas capas en GPU con deteccion de cliff (dense ~42%, MoE ~15%)
- **Thread tuning** — 8 hilos (dense), 6 hilos (MoE)
- **mlock** — Bloquea modelo en RAM
- **--no-mmap** — Auto-activado para MoE (+28% velocidad)
- **q4_0 KV cache** — Cache de atencion comprimido
- **Deteccion MoE** — Regex automatico en nombre del modelo

## Detener Servidor

```bash
# Linux
bash scripts/kill-server.sh

# Windows
scripts\kill-server.bat

# macOS
bash scripts/kill-server-mac.sh
```

Solo detiene QuantumLeap. Ollama no se toca.

## Re-setup en Otro Equipo

```bash
cd ~/Documents/Proyectos/QuantumLeap
bash scripts/setup.sh    # detecta hardware nuevo automaticamente
bash scripts/start.sh    # listo
```

El setup es idempotente — se puede ejecutar multiples veces sin problemas.

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| Engine no construido | `bash scripts/setup.sh` |
| GPU no detectada | Normal en CPU-only, funciona sin GPU |
| Modelos no aparecen | Verificar `ls models/*.gguf`, reiniciar servidor |
| Puerto ocupado | `API_PORT=11436 bash scripts/start.sh` |
| CUDA no encontrado | Instalar CUDA Toolkit, re-ejecutar setup |
| Python < 3.10 | `python3.10 -m venv .venv && source .venv/bin/activate` |

## Estructura del Proyecto

```
QuantumLeap/
├── engine/llama.cpp/       # Backend de inferencia (ik_llama.cpp)
├── core/                   # TurboQuant C++/CUDA/ASM
│   ├── include/turboquant/ # Headers
│   ├── src/turboquant/     # Implementacion + kernels CUDA
│   └── tests/              # Tests unitarios + benchmarks
├── api/server.py           # FastAPI (Ollama + OpenAI API)
├── web/                    # Web UI
├── models/                 # Archivos GGUF
├── scripts/                # Setup, start, stop (cross-platform)
└── config.json             # Configuracion persistente
```

## Documentacion

- **[README.md](README.md)** — Documentacion principal
- **[QUICKSTART.md](QUICKSTART.md)** — Guia rapida (5 min)
- **[CHANGELOG.md](CHANGELOG.md)** — Historial de versiones
