$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep_2"
$pythonPath = "C:\CODE2026\.venv\Scripts\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Starting Yamawaki_Rep_2 Orchestrator (100% Replication Pipeline)   " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Set-Location -Path "C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep_2"

# Step 1: Run 100-Point Audit
Write-Host "[*] Executing 100-Point Pre-flight Audit Rubric..." -ForegroundColor Yellow
& "$pythonPath" audit_rubric.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Pre-flight audit failed. Aborting training to prevent invalid execution." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Audit passed with 100/100 points! Starting training session..." -ForegroundColor Green

# Step 2: Launch Training with AMP, ETA and Dashboards
$epochs = 400
$batchSize = 16
$lr = 0.001

Write-Host "[*] Launching training for $epochs epochs (Batch Size: $batchSize, LR: $lr)..." -ForegroundColor Yellow
& "$pythonPath" train.py --epochs $epochs --batch_size $batchSize --lr $lr

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Training error encountered. Exiting orchestrator." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[+] Training completed successfully!" -ForegroundColor Green

# Step 3: Run Full Evaluation across 20 Test Scenes
Write-Host "[*] Running final quantitative benchmark evaluation on 20 test scenes..." -ForegroundColor Yellow
& "$pythonPath" evaluate.py

# Step 4: Final Git Push
Write-Host "[-] Committing and pushing final results to GitHub..." -ForegroundColor Cyan
git add results/ README.md
git commit -m "Results: Completed 400 epochs training and 20-scene benchmark evaluation"
git push origin main

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   Replication Pipeline Finished and Synced to GitHub Successfully!   " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
