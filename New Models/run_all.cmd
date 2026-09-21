@echo off
cd /d "%~dp0"
del gcn_done.marker tcn_done.marker 2>nul
python -u GCN\train_gcn.py ..\..\X_195_75.npy ..\..\y_195_75.npy > gcn_wmi.log 2>&1
echo EXITCODE %ERRORLEVEL% > gcn_done.marker
python -u TCN\train_tcn.py ..\..\X_195_75.npy ..\..\y_195_75.npy > tcn_wmi.log 2>&1
echo EXITCODE %ERRORLEVEL% > tcn_done.marker
