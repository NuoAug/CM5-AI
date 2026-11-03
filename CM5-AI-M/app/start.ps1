# Windows entry for the Raspberry Pi test bench: view only.
param([int]$LocalPort=18090)
$ErrorActionPreference='Stop'
$client=New-Object System.Net.Sockets.TcpClient
try { $client.Connect('127.0.0.1',$LocalPort) }
catch { throw 'SSH tunnel is not running. See the hardware test guide in the workspace root.' }
finally { $client.Close() }
Start-Process "http://127.0.0.1:$LocalPort/"
