# Stops the running G3 chain (frederik/g3_chain.ps1, or a G3 script started on its own) together with every process
# it started, nnU-Net's data workers included. Finished work is kept: started again with frederik\g3_start.ps1, a stage
# skips the finished runs and evaluated scans, and a training run continues from its last finished epoch
# (scripts/g3_train.py). Only the epoch or the scan in progress is lost.
#
#   powershell -ExecutionPolicy Bypass -File frederik\g3_stop.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)
$found = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(powershell|pwsh|cmd|python)\.exe$' -and
    $_.CommandLine -match 'g3_chain\.ps1|g3_(run|prepare|train|evaluate)\.py' }
if (-not $found) {
    Write-Output "No G3 chain is running."
    exit 0
}
# The chain first, so it cannot start a next step; /T ends each process with all its child processes.
foreach ($p in $found | Sort-Object { $_.CommandLine -notmatch 'g3_chain\.ps1' }) {
    taskkill /PID $p.ProcessId /T /F *> $null
}
$status = "data\g3_kits\logs\chain_status.txt"
if (Test-Path (Split-Path $status)) {
    Add-Content $status "$(Get-Date -Format s) STOPPED (g3_stop.ps1)"
}
Write-Output "Stopped G3 ($(@($found).Count) processes). To continue: powershell -ExecutionPolicy Bypass -File frederik\g3_start.ps1"
