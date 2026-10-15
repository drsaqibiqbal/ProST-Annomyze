@echo off
REM Launch the ProST-Annomyze v2 dashboard and open it in the browser.
REM Uses this folder's src\ so it always runs v2's code (v1 stays on port 8503).
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "KMP_DUPLICATE_LIB_OK=TRUE"
REM Open the browser a few seconds after the server starts.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8504"
python -m streamlit run dashboard\convert_app.py --server.port 8504 --server.headless true --browser.gatherUsageStats false
pause
