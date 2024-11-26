@echo off
echo Setting up R dependencies for BioNeuralNet...

:: Check if R is installed
where R >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo R is not installed. Installing R...
    powershell -Command "Invoke-WebRequest -Uri 'https://cran.r-project.org/bin/windows/base/R-4.3.1-win.exe' -OutFile '%TEMP%\R-installer.exe'"
    start /wait %TEMP%\R-installer.exe /SILENT
) else (
    echo R is already installed.
)

:: Install R packages
echo Installing R packages: dplyr, SmCCNet, WGCNA...
Rscript -e "install.packages(c('dplyr', 'SmCCNet', 'WGCNA'), repos='http://cran.r-project.org')"

echo R dependencies setup completed!
pause
