@echo off
REM Stop QuantumLeap server (does NOT touch Ollama on port 11434)
if "%API_PORT%"=="" set API_PORT=11435

echo Stopping QuantumLeap on port %API_PORT%...

REM Kill processes on QuantumLeap port
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%API_PORT% ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    echo   stopped API on :%API_PORT% (PID %%a)
)

REM Kill internal llama-server on port 8081
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    echo   stopped llama-server on :8081 (PID %%a)
)

timeout /t 2 /nobreak >nul

REM Verify
netstat -ano | findstr :%API_PORT% | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo   [!] Port %API_PORT% still in use
    exit /b 1
) else (
    echo   Port %API_PORT% is free
)

echo Done (Ollama on :11434 untouched)
