# Starts G3 stages (frederik/g3_chain.ps1) as a process of its own: it keeps running when this window, VS Code or a
# Claude session is closed, but not when the PC is shut down. Every stage skips finished work, so this is also how a
# stopped chain is continued. Without stages it runs the learning curves.
#
#   powershell -ExecutionPolicy Bypass -File frederik\g3_start.ps1               # = curves
#   powershell -ExecutionPolicy Bypass -File frederik\g3_start.ps1 grid curves

param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Stages)
if (-not $Stages) { $Stages = @("curves") }
$repo = Split-Path $PSScriptRoot -Parent
# One chain at a time: two would train and evaluate the same runs at once.
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(powershell|pwsh|cmd|python)\.exe$' -and
    $_.CommandLine -match 'g3_chain\.ps1|g3_(run|prepare|train|evaluate)\.py' }
if ($running) {
    Write-Output "G3 is already running (PID $(@($running.ProcessId) -join ', ')). To restart it, stop it first: frederik\g3_stop.ps1"
    exit 1
}
$cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$repo\frederik\g3_chain.ps1`" $($Stages -join ' ')"
$started = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo }
if ($started.ReturnValue -ne 0) {
    Write-Output "Could not start G3 (Win32_Process.Create returned $($started.ReturnValue))."
    exit 1
}
Write-Output "Started G3: $($Stages -join ', ') (PID $($started.ProcessId)). Progress: data\g3_kits\logs\chain_status.txt; stop: frederik\g3_stop.ps1"
