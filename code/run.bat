@echo off

cd /e "%~dp0"

call activate pytorch

start call python run.py --data_index 0

exit
