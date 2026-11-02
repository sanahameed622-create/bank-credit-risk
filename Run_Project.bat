@echo off
title Bank Credit Risk & Scorecard Engine
color 0B
echo ================================================================
echo          BANK CREDIT RISK SCORING MODEL (PROJECT 1)
echo ================================================================
echo.
cd /d "%~dp0"
python credit_scorecard.py
echo.
echo Launching Interactive Visual Dashboard in your browser...
start "" credit_score_dashboard.html
echo.
echo ================================================================
echo Press any key to test Custom Live Applicant...
echo ================================================================
pause >nul
python interactive_evaluator.py
pause
