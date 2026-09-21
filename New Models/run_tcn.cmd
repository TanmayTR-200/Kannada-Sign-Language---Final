@echo off
cd /d "%~dp0"
python -u TCN\train_tcn.py ..\..\X_195_75.npy ..\..\y_195_75.npy > tcn_wmi.log 2>&1
echo EXITCODE %ERRORLEVEL% > tcn_done.marker
