# Runs G3 stages one after another (Frederik's Windows PC; the stages are scripts/g3_run.py's: prepare, test, grid,
# curves). It stops at the first stage that fails. Every stage skips finished work, so the same command can simply be
# started again. Usually started with frederik\g3_start.ps1, which runs it as a process of its own (independent of
# any terminal or Claude session), and stopped with frederik\g3_stop.ps1.
#
#   powershell -ExecutionPolicy Bypass -File frederik\g3_chain.ps1 prepare test grid
#
# Progress: data\g3_kits\logs\chain_status.txt (one line per stage start and end), chain_<stage>.out (what g3_run.py
# prints: one line per step) and <stage>.log (the full output of every step).

param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Stages)

Set-Location (Split-Path $PSScriptRoot -Parent)
# One chain at a time: two would train and evaluate the same runs at once.
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(powershell|pwsh|cmd|python)\.exe$' -and
    $_.CommandLine -match 'g3_chain\.ps1|g3_(run|prepare|train|evaluate)\.py' }
if ($running) {
    Write-Output "G3 is already running (PID $(@($running.ProcessId) -join ', ')); stop it first: frederik\g3_stop.ps1"
    exit 1
}
$env:PYTHONIOENCODING = "utf-8"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $env:Path = "$env:ProgramFiles\Git\cmd;$env:Path"      # so every run can record its git commit (machine.json)
}
$logs = "data\g3_kits\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$status = Join-Path $logs "chain_status.txt"

foreach ($stage in $Stages) {
    Add-Content $status "$(Get-Date -Format s) START $stage"
    # cmd's redirection keeps Python's UTF-8 output as it is (PowerShell 5.1 would rewrite it as UTF-16).
    cmd /c ".venv\Scripts\python.exe -u scripts\g3_run.py --config configs\g3_kits.yaml --stage $stage >> $logs\chain_$stage.out 2>&1"
    $code = $LASTEXITCODE
    Add-Content $status "$(Get-Date -Format s) END $stage exit $code"
    if ($code -ne 0) {
        Add-Content $status "$(Get-Date -Format s) STOPPED"
        exit $code
    }
}
Add-Content $status "$(Get-Date -Format s) DONE"
