@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 20844)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 25836)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 12268)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 14680)

del /F cleanup-ansys-DESKTOP-Q32N9SO-14680.bat
