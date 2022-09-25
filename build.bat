@echo off
for /F "delims=#" %%E in ('"prompt #$E# & for %%E in (1) do rem"') do set "ESCchar=%%E"
set "white=%ESCchar%[90m"
set "red=%ESCchar%[91m"
set "green=%ESCchar%[92m"
set "yellow=%ESCchar%[93m"
set "blue=%ESCchar%[94m"
set "magenta=%ESCchar%[95m"
set "cyan=%ESCchar%[96m"
set "endcol=%ESCchar%[0m"

IF NOT EXIST build mkdir build 
pushd build

del *.pdb > NUL 2> NUL

echo:
set STARTTIME=%TIME%

echo Build date: %DATE%

if "%1" == "--datamovement"  ( nvcc ../datamovement/datamovement.cu -o datamovement.exe
) else if "%1" == "--reduce" ( nvcc ../reduce/reduce.cu -o reduce.exe
) else if "%1" == "--scan"   ( nvcc ../scan/scan.cu -o scan.exe

) else ( echo %red%Invalid build command!%endcol% ) 

echo:
echo Build start: %STARTTIME%
echo Build ready: %TIME%
popd
