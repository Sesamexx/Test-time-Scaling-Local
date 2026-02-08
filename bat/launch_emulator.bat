@echo off
REM ============================================================================
REM Launch Android Emulator for AndroidWorld Benchmark
REM ============================================================================

REM Set your Android SDK path here
REM Common locations:
REM   C:\Users\<USERNAME>\AppData\Local\Android\Sdk
REM   C:\Android\Sdk

set ANDROID_SDK=%LOCALAPPDATA%\Android\Sdk

REM Check if emulator exists
if not exist "%ANDROID_SDK%\emulator\emulator.exe" (
    echo ERROR: Emulator not found at %ANDROID_SDK%\emulator\emulator.exe
    echo Please update ANDROID_SDK path in this script.
    echo.
    echo Your Android SDK is likely at one of:
    echo   - %LOCALAPPDATA%\Android\Sdk
    echo   - C:\Android\Sdk
    echo.
    echo Check Android Studio: Tools -^> SDK Manager -^> Android SDK Location
    pause
    exit /b 1
)

echo ============================================================================
echo Launching Android Emulator for AndroidWorld
echo ============================================================================
echo.
echo AVD Name: AndroidWorldAvd
echo GRPC Port: 8554
echo.
echo NOTE: If AVD not found, create it in Android Studio:
echo   Tools -^> Device Manager -^> Create Device
echo   Hardware: Pixel 6
echo   System Image: Tiramisu (API 33)
echo   AVD Name: AndroidWorldAvd
echo.
echo ============================================================================
echo.

"%ANDROID_SDK%\emulator\emulator.exe" -avd AndroidWorldAvd -no-snapshot -grpc 8554

pause
