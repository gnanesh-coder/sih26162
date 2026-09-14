@echo off
REM Starts the SIH26162 dashboard against the full 12-month national archive.
REM
REM PROCESSED_CORPUS defaults to whatever a short pipeline run last wrote, which
REM is a 2-day file. Every endpoint then serves a plausible-looking answer built
REM from 0.07%% of the data -- the compliance register once reported 465 routine
REM detections instead of 485,781 and said nothing about it. Start the demo from
REM here rather than from a bare uvicorn command.
setlocal
if "%PROCESSED_CORPUS%"=="" set "PROCESSED_CORPUS=data/processed/firms_industrial_joined_12m.parquet"
if "%PORT%"=="" set "PORT=8077"
echo Corpus : %PROCESSED_CORPUS%
echo Port   : %PORT%
".venv\Scripts\python.exe" -m uvicorn app.main:app --port %PORT% %*
