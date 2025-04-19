@echo off
echo ===============================
echo Launching Multi-Agent System...
echo ===============================

REM --- Extraction Agent ---
start cmd /k "cd Extraction_agent && uvicorn main:app --host localhost --port 8001 --reload"

REM --- Similarity Agent ---
start cmd /k "cd similarity_agent && uvicorn main1:app --host localhost --port 8004 --reload"

REM --- Relation Agent ---
start cmd /k "cd reasoning_agent_relation && uvicorn main2:app --host localhost --port 8005 --reload"

REM --- Coordinator Agent ---
start cmd /k "cd coordinator && uvicorn main:app --host localhost --port 8006 --reload"

echo All agents started.
