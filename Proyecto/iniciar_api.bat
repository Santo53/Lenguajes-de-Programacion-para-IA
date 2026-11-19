@echo off
REM API de precios STB (modelo v2, solo texto). Doble clic para arrancar; cerrar la ventana para detener.
cd /d "%~dp0"
title API de precios STB
echo.
echo  Arrancando la API de precios (modelo v2)...
echo  Documentacion y pruebas:  http://127.0.0.1:8000/docs
echo.
"..\.venv\Scripts\python.exe" src\main.py
pause
