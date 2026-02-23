@echo off
echo.
echo ============================================================
echo Starting SSH Tunnel
echo ============================================================
echo Local Port: 9000
echo Remote: ubuntu@36.103.234.60:80
echo.
echo The tunnel will stay alive until you close this window
echo Press Ctrl+C to stop the tunnel
echo ============================================================
echo.

REM Start SSH tunnel with keepalive options
ssh -L 9000:localhost:80 ^
    -o ServerAliveInterval=60 ^
    -o ServerAliveCountMax=3000 ^
    -o TCPKeepAlive=yes ^
    -N ^
    ubuntu@36.103.234.60

echo.
echo Tunnel closed.
pause
