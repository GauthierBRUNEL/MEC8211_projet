@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 29780)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 13984)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 7956)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 17384)

del /F cleanup-ansys-DESKTOP-Q32N9SO-17384.bat
