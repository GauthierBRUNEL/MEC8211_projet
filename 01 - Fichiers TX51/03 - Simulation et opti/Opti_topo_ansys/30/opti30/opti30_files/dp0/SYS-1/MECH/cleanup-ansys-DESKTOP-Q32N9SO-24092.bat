@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 23304)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 27036)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 31828)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 24092)

del /F cleanup-ansys-DESKTOP-Q32N9SO-24092.bat
