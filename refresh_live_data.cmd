@echo off
REM Pull the last 2 days of NASA FIRMS, run the pipeline, reseed the database.
REM
REM The dashboard's freshness badge is computed from the newest detection it
REM holds, so it will read STALE until this is run. FIRMS NRT publishes about
REM 3 hours behind the overpass; run this shortly before any demo.
setlocal
echo [1/3] Pulling FIRMS NRT (VIIRS S-NPP, NOAA-20, MODIS)...
".venv\Scripts\python.exe" -c "import pandas as pd, geopandas as gpd; from src.ingestion.firms_client import fetch_firms_nrt; fr=[g for s in ('VIIRS_SNPP_NRT','VIIRS_NOAA20_NRT','MODIS_NRT') for g in [fetch_firms_nrt(satellite=s, day_range=2)] if g is not None and len(g)]; c=gpd.GeoDataFrame(pd.concat(fr, ignore_index=True), crs=fr[0].crs).drop_duplicates(subset=['latitude','longitude','acq_date','acq_time','satellite']); c.to_parquet('data/raw/firms_latest.parquet', index=False); print('   ->', len(c), 'detections,', c.acq_date.min(), 'to', c.acq_date.max())"
if errorlevel 1 goto :fail

echo [2/3] Spatial join, recurrence state machine, alert tiers...
".venv\Scripts\python.exe" -m src.pipeline.spatial_join --firms data/raw/firms_latest.parquet --output data/processed/firms_industrial_joined.parquet
if errorlevel 1 goto :fail

echo [3/3] Reseeding the incident database...
".venv\Scripts\python.exe" -c "from app.database import SessionLocal, init_db, Incident; from app.main import seed_database_from_parquet; from src.pipeline.spatial_join import OUTPUT_PROCESSED_PARQUET; init_db(); db=SessionLocal(); n=seed_database_from_parquet(db, OUTPUT_PROCESSED_PARQUET); print('   ->', n, 'new,', db.query(Incident).count(), 'total')"
if errorlevel 1 goto :fail

echo.
echo Done. Restart the dashboard (run_dashboard.cmd) to pick up the new data.
goto :eof

:fail
echo.
echo FAILED. The dashboard will keep serving whatever it already had, and its
echo freshness badge will say so rather than claiming LIVE.
exit /b 1
