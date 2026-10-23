@echo off
cd /d "%~dp0"
echo Starting AURALIS from: %cd%
echo Using fresh port 8502 to avoid the old blank Streamlit session.
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.port 8502
) else (
    python -m streamlit run "%~dp0app.py" --server.port 8502
)
pause
