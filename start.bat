@echo off
setlocal

:: Kill any old HTTP servers on port 8000 so we get a clean start
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 "') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Start the Python HTTP server in the background for the frontend UI
echo Starting UI server on http://localhost:8000
start "Deriv Trading UI" /b cmd /c "cd frontend && python -m http.server 8000 >nul 2>&1"

:: Wait a moment for the server to start
timeout /t 2 /nobreak >nul

:: Open the dashboard in the default browser
echo Opening dashboard at http://localhost:8000
start "" "http://localhost:8000"

:: Start the main trading bot
echo Starting Deriv Trading Bot...
echo.
python main.py %*

:: Cleanup: when the bot stops, kill the background web server
taskkill /f /im python.exe /fi "WINDOWTITLE eq Deriv Trading UI*" >nul 2>&1
