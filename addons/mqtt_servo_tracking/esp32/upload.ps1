# Upload face_tracker to ESP32 Dev Module @ 115200 baud (more reliable on Windows).
param(
    [string]$Port = "COM5",
    [string]$Fqbn = "esp32:esp32:esp32",
    [int]$Baud = 115200
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$SketchDir = Join-Path $PSScriptRoot "face_tracker"
$CliCandidates = @(
    "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
    "arduino-cli"
)

$Cli = $CliCandidates | Where-Object { Test-Path $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $Cli) {
    $Cli = (Get-Command arduino-cli -ErrorAction SilentlyContinue).Source
}
if (-not $Cli) {
    throw "arduino-cli not found. Install Arduino IDE or add arduino-cli to PATH."
}

Write-Host "[upload] Port=$Port FQBN=$Fqbn Baud=$Baud"
Write-Host ""
Write-Host "BEFORE upload:"
Write-Host "  1. Close Arduino Serial Monitor and any app using $Port"
Write-Host "  2. Disconnect servo power + signal wire (servo can block boot)"
Write-Host "  3. Use a data USB cable (not charge-only)"
Write-Host ""
Write-Host "WHEN you see Connecting....:"
Write-Host "  Hold BOOT -> tap EN/RESET -> release EN -> release BOOT after 2s"
Write-Host ""

& $Cli compile --fqbn $Fqbn --build-property "upload.speed=$Baud" $SketchDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Cli upload -p $Port --fqbn $Fqbn `
    --upload-property "upload.speed=$Baud" `
    --upload-property "serial.disableDTR=false" `
    --upload-property "serial.disableRTS=false" `
    $SketchDir
exit $LASTEXITCODE
