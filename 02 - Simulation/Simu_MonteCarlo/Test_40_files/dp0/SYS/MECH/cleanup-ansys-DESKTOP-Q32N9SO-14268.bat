@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 17712)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 7916)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 21380)
if /i "%LOCALHOST%"=="DESKTOP-Q32N9SO" (taskkill /f /pid 14268)

del /F cleanup-ansys-DESKTOP-Q32N9SO-14268.bat
