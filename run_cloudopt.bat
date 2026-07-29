 @echo off
REM CloudOpt Project Launcher
REM This script starts ML service, backend, and frontend in separate terminals.

REM Start ML Service
start "ML Service" cmd /k "cd ml-service && py -m uvicorn main:app --host 0.0.0.0 --port 8000"

REM Start Backend
start "Backend" cmd /k "cd backend && mvn spring-boot:run"

REM Start Frontend
start "Frontend" cmd /k "cd frontend && npm start"

echo All services are launching in separate windows.
pause
