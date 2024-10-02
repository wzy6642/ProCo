@echo off

cd /e "%~dp0"

call activate pytorch

python score.py --data_index 0

pause
