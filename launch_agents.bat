@echo off
echo ===============================
echo Launching Multi-Agent System...
echo ===============================

REM --- Similarity Agent ---
start cmd /k "cd similarity_agent && uvicorn main1:app --host localhost --port 8004 --reload"

REM --- Relation Agent ---
start cmd /k "cd reasoning_agent_relation && uvicorn main2:app --host localhost --port 8005 --reload"

REM --- Coordinator Agent ---
start cmd /k "cd coordinator && uvicorn main:app --host localhost --port 8006 --reload"

REM --- Goal Prediction Agent ---
start cmd /k "cd goal_prediction_agent && uvicorn main:app --host localhost --port 8007 --reload"

REM --- Contradiction Detection Agent ---
start cmd /k "cd contradiction_agent && uvicorn main:app --host localhost --port 8008 --reload"

REM --- Critic Agent
start cmd /k "cd critic_agent && uvicorn main:app --host localhost --port 8009 --reload"

REM --- Verifier Agent ---
start cmd /k "cd verifier_agent && uvicorn main:app --host localhost --port 8010 --reload"





echo All agents started.
pause
