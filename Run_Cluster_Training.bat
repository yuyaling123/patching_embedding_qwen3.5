@echo off
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%PYTHONPATH%;.

echo ========================================================
echo Starting Cluster-Based Time-LLM Training
echo This script will:
echo 1. Split data into 3 clusters based on station_to_cluster.json
echo 2. Train a separate Time-LLM model for each cluster
echo ========================================================

:: Use the specific python environment
D:\Anaconda3\envs\timellm\python.exe scripts/run_all_clusters.py

echo.
echo ========================================================
echo All Clusters Processed.
echo ========================================================
pause
