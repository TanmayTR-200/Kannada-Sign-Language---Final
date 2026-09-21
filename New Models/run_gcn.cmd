@echo off
cd /d "%~dp0"
python -u GCN\train_gcn.py ..\..\X_195_75.npy ..\..\y_195_75.npy > gcn_wmi.log 2>&1
echo EXITCODE %ERRORLEVEL% > gcn_done.marker
