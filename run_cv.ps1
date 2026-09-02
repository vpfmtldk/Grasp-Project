# K-fold (image-wise) cross-validation for GG-CNN on Cornell.
#
# For each fold it rotates the dataset (--ds-rotate), trains on the first 90%,
# then evaluates on the held-out last 10% (the split eval_ggcnn uses with the
# same --ds-rotate is exactly the data NOT seen in training).
#
# Sequential + unattended. Each fold's training goes through run_train.ps1, so
# crash/hang auto-resume applies and an interrupted CV run continues where it
# stopped (re-run the same command).
#
# Examples:
#   .\run_cv.ps1                       # RGB-only GG-CNN, 25 epochs, 5 folds
#   .\run_cv.ps1 -UseRgb 0 -UseDepth 1 # depth GG-CNN
#   .\run_cv.ps1 -Network ggcnn2 -Epochs 30

param(
    [string]   $Network     = 'ggcnn',
    [int]      $UseRgb       = 1,
    [int]      $UseDepth     = 0,
    [int]      $Epochs       = 25,
    [double[]] $Folds        = @(0.0, 0.2, 0.4, 0.6, 0.8),
    [int]      $NumWorkers   = 0,
    [string]   $DatasetPath  = 'C:\Users\135\Downloads\cornell_grasp'
)

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

$python  = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$tag     = "$Network`_rgb$UseRgb`_d$UseDepth`_e$Epochs"
$results = Join-Path $PSScriptRoot "output\cv_${tag}.txt"
"# cross-validation  $tag  started $(Get-Date -Format s)" | Out-File $results -Encoding utf8

foreach ($f in $Folds) {
    $fs     = ([string]$f) -replace '\.', 'p'
    $runDir = Join-Path $PSScriptRoot "output\models\cv_${tag}_ds${fs}"

    Write-Host ""
    Write-Host "############ FOLD ds-rotate=$f  ->  $runDir ############" -ForegroundColor Magenta

    & (Join-Path $PSScriptRoot 'run_train.ps1') `
        -Network $Network -RunDir $runDir `
        -UseRgb $UseRgb -UseDepth $UseDepth -DsRotate $f `
        -Epochs $Epochs -NumWorkers $NumWorkers -DatasetPath $DatasetPath

    # best-IoU checkpoint for this fold
    $ckpts = Get-ChildItem "$runDir\epoch_*_statedict.pt" -ErrorAction SilentlyContinue
    if (-not $ckpts) {
        "ds=$f  ERROR: no checkpoint produced" | Tee-Object -FilePath $results -Append
        continue
    }
    $best = $ckpts | Sort-Object { [double]($_.Name -replace '.*_iou_([0-9.]+)_statedict\.pt', '$1') } |
            Select-Object -Last 1
    $netArg = $best.FullName -replace '_statedict\.pt$', ''

    Write-Host "eval: $netArg" -ForegroundColor Magenta
    $evalOut = & $python eval_ggcnn.py --network $netArg --dataset cornell `
        --dataset-path $DatasetPath --use-rgb $UseRgb --use-depth $UseDepth `
        --ds-rotate $f --iou-eval --num-workers $NumWorkers 2>&1
    $line = ($evalOut | Select-String 'IOU Results').Line
    "ds=$f  best=$($best.Name)  $line" | Tee-Object -FilePath $results -Append
}

# aggregate
$vals = Select-String -Path $results -Pattern '= ([0-9.]+)\s*$' |
        ForEach-Object { [double]$_.Matches[0].Groups[1].Value }
if ($vals.Count) {
    $mean = ($vals | Measure-Object -Average).Average
    $sd   = [math]::Sqrt((($vals | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) | Measure-Object -Sum).Sum / $vals.Count)
    ("`n{0} folds  mean IoU = {1:N4}  sd = {2:N4}  ({3})" -f $vals.Count, $mean, $sd,
        (($vals | ForEach-Object { '{0:N3}' -f $_ }) -join ', ')) | Tee-Object -FilePath $results -Append
}

Write-Host "`n############ CV DONE ############" -ForegroundColor Green
Get-Content $results
