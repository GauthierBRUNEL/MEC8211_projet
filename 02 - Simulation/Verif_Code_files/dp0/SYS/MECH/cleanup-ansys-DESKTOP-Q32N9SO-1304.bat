@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 21752)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 11096)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 5376)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 1304)

del /F cleanup-ansys-DESKTOP-Q32N9SO-1304.bat
