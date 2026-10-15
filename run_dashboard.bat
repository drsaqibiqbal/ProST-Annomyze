@echo off
REM Launch the ProST-Annomyze dashboard (v3) on port 8520 and open it in the browser.
REM Uses this folder's src\ so it always runs this version's code.
REM Other devices on the same network: http://<this-PC's-IP>:8520
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "KMP_DUPLICATE_LIB_OK=TRUE"
REM Open the browser a few seconds after the server starts.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8520"
python -m streamlit run dashboard\convert_app.py --server.port 8520 --server.headless true --browser.gatherUsageStats false
pause
