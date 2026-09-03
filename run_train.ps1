# GG-CNN training with automatic crash-AND-hang resume.
#
# Usage examples (from the repo root):
#   .\run_train.ps1                                   # GG-CNN2, depth-only, new run folder
#   .\run_train.ps1 -Network ggcnn -UseRgb 1          # GG-CNN, RGB-D
#   .\run_train.ps1 -RunDir output\models\260901_1924_training_example -Network ggcnn   # resume a specific run
#
# Behaviour:
#   - resumes from "<RunDir>\ckpt_last.pt" (model + optimizer + epoch) if present,
#     else the newest "epoch_*_statedict.pt" (weights only, next epoch);
#   - restarts on crash (non-zero exit);
#   - WATCHDOG: also restarts if training hangs (ckpt_last.pt not updated for $staleMinutes;
#     Windows DataLoader deadlocks freeze instead of crashing);
#   - stops on normal completion (exit 0) or Ctrl+C.

param(
    [string] $Network      = 'ggcnn2',
    [string] $RunDir       = '',
    [string] $DatasetPath  = 'C:\Users\135\Downloads\cornell_grasp',
    [string] $Dataset      = 'cornell',
    [int]    $UseRgb       = 0,
    [int]    $UseDepth     = 1,
    [double] $DsRotate     = 0.0,
    [int]    $Epochs       = 50,
    [int]    $NumWorkers   = 0,          # 0 = safest on Windows
    [int]    $StaleMinutes = 20,
    [int]    $MaxRetries   = 1000,
    [int]    $RetryWait    = 10
)

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$logDir = Join-Path $PSScriptRoot 'output'

$desc = "$Network`_rgb$UseRgb`_d$UseDepth"
if (-not $RunDir) {
    $stamp = Get-Date -Format 'yyMMdd_HHmm'
    $RunDir = Join-Path $PSScriptRoot "output\models\${stamp}_${desc}"
}
if (-not (Test-Path $RunDir)) { New-Item -ItemType Directory -Path $RunDir -Force | Out-Null }
Write-Host "run dir: $RunDir" -ForegroundColor Green

$commonArgs = @(
    'train_ggcnn.py',
    '--description',  $desc,
    '--network',      $Network,
    '--dataset',      $Dataset,
    '--dataset-path', $DatasetPath,
    '--use-rgb',      "$UseRgb",
    '--use-depth',    "$UseDepth",
    '--ds-rotate',    "$DsRotate",
    '--num-workers',  "$NumWorkers",
    '--epochs',       "$Epochs",
    '--save-folder',  $RunDir           # train_ggcnn.py writes here and auto-resumes from RunDir\ckpt_last.pt
)

function Get-ResumeArgs {
    $ckptLast = Join-Path $RunDir 'ckpt_last.pt'
    if (Test-Path $ckptLast) { return @('--resume', $ckptLast) }
    $sd = Get-ChildItem -Path $RunDir -Filter 'epoch_*_statedict.pt' -ErrorAction SilentlyContinue |
          Sort-Object Name | Select-Object -Last 1
    if ($sd) {
        $e = [int]($sd.Name -replace '^epoch_(\d+).*$', '$1')
        return @('--resume', $sd.FullName, '--start-epoch', "$($e + 1)")
    }
    return @()
}

for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {

    $resumeArgs = Get-ResumeArgs
    $stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
    $outLog = Join-Path $logDir "train_$stamp.out.log"
    $errLog = Join-Path $logDir "train_$stamp.err.log"

    Write-Host ""
    Write-Host "=== [attempt $attempt] $(Get-Date -Format s) | $desc ===" -ForegroundColor Cyan
    if ($resumeArgs.Count) { Write-Host "resume: $($resumeArgs -join ' ')" -ForegroundColor Cyan }
    else                   { Write-Host "resume: (fresh start)"           -ForegroundColor Cyan }
    Write-Host "log:    $errLog" -ForegroundColor DarkGray

    $proc = Start-Process -FilePath $python -ArgumentList ($commonArgs + $resumeArgs) `
                          -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru `
                          -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    $ckptLast = Join-Path $RunDir 'ckpt_last.pt'
    $hung = $false

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 30

        $ref = $null
        if (Test-Path $ckptLast) { $ref = (Get-Item $ckptLast).LastWriteTime }
        if (Test-Path $errLog) {
            $le = (Get-Item $errLog).LastWriteTime
            if (-not $ref -or $le -gt $ref) { $ref = $le }
        }
        if (-not $ref) { $ref = $proc.StartTime }

        if (((Get-Date) - $ref).TotalMinutes -ge $StaleMinutes) {
            Write-Host ("=== HANG: no progress {0:N1} min. Killing PID {1} ===" -f ((Get-Date) - $ref).TotalMinutes, $proc.Id) -ForegroundColor Red
            taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            $hung = $true
            Start-Sleep -Seconds 5
            break
        }
    }

    if (-not $hung) {
        $proc.WaitForExit()          # ensures ExitCode is populated (PS quirk with -PassThru)
        $code = $proc.ExitCode
        if ($null -eq $code) { $code = 0 }
        if ($code -eq 0) {
            Write-Host "=== done (exit 0) ===" -ForegroundColor Green
            break
        }
        Write-Host "=== crashed (exit $code) ===" -ForegroundColor Yellow
    }

    Write-Host "=== restarting in ${RetryWait}s (Ctrl+C to stop) ===" -ForegroundColor Yellow
    Start-Sleep -Seconds $RetryWait
}
