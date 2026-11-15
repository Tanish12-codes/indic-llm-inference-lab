# Run one step (or all) and save a log to results\logs\.
#   .\run.ps1 env      environment check
#   .\run.ps1 1        token-count study
#   .\run.ps1 tests    correctness tests (must pass before any speed measurement)
#   .\run.ps1 2 ... 9  experiments
#   .\run.ps1 all      tests, then 2..9 (stops if tests fail)
# If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File .\run.ps1 1
param([Parameter(Mandatory = $true)][string]$step)

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # show Devanagari correctly
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No .venv found - run .\setup_env.ps1 first"; exit 1 }

$exp = @{
    "1" = "experiments\01_token_counts\run.py"
    "2" = "experiments\02_gpt2_scratch\run.py"
    "3" = "experiments\03_kv_cache\run.py"
    "4" = "experiments\04_speed_memory\run.py"
    "5" = "experiments\05_prealloc_cache\run.py"
    "6" = "experiments\06_int8_kv_cache\run.py"
    "7" = "experiments\07_mha_gqa_mqa\run.py"
    "8" = "experiments\08_memory_calculator\run.py"
    "9" = "experiments\09_combine_language_kv\run.py"
}

function Run-One([string]$s) {
    if ($s -eq "env") {
        & $py scripts\run_logged.py env $py scripts\env_check.py | Out-Host
    } elseif ($s -eq "tests") {
        & $py scripts\run_logged.py tests $py -m pytest tests -v -s | Out-Host
    } elseif ($exp.ContainsKey($s)) {
        & $py scripts\run_logged.py ("step" + $s) $py $exp[$s] | Out-Host
    } else {
        Write-Error "unknown step '$s'"; exit 1
    }
    return $LASTEXITCODE
}

if ($step -eq "all") {
    if ((Run-One "tests") -ne 0) { Write-Host "Tests failed - not measuring speed."; exit 1 }
    foreach ($s in "2", "3", "4", "5", "6", "7", "8", "9") {
        if ((Run-One $s) -ne 0) { Write-Host "Step $s failed - stopping."; exit 1 }
    }
} else {
    exit (Run-One $step)
}
