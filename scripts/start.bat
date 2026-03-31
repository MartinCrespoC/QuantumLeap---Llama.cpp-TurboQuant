@echo off
REM QuantumLeap — Optimized LLM Inference (coexists with Ollama)
REM Windows start script. Requires: Python 3.10+, CUDA Toolkit (optional)

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
if "%API_PORT%"=="" set API_PORT=11435

cd /d "%PROJECT_ROOT%"

echo.
echo   QuantumLeap v0.5.0 — Optimized LLM Inference
echo   Built on llama.cpp ^| TurboQuant Engine
echo.

REM Check engine
if not exist "engine\llama.cpp\build\bin\llama-server.exe" (
    if not exist "engine\llama.cpp\build\bin\Release\llama-server.exe" (
        echo   [!] Engine not built. Run: scripts\setup.bat
        exit /b 1
    )
)

REM Detect GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo   GPU: %%a
) else (
    echo   GPU: Not detected (CPU-only mode^)
)

REM Python venv
if not exist ".venv" if not exist "venv" (
    echo   Creating Python virtual environment...
    python -m venv .venv
)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python -c "import fastapi" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   Installing Python dependencies...
    pip install -q -r api\requirements.txt
)

REM Stop any previous QuantumLeap on our port (NOT Ollama on 11434)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%API_PORT% ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul

echo.
echo   Starting QuantumLeap...
echo.
echo   Web UI:     http://localhost:%API_PORT%
echo   Ollama API: http://localhost:%API_PORT%/api/
echo   OpenAI API: http://localhost:%API_PORT%/v1/
echo.
echo   Ollama coexistence: Ollama stays on :11434, QuantumLeap on :%API_PORT%
echo   To replace Ollama:  set API_PORT=11434 ^& scripts\start.bat
echo.
echo   Press Ctrl+C to stop
echo.

python api\server.py %*
