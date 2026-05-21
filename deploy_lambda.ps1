# TOMLSignals - Package & Deploy to Lambda Labs
# Usage: .\deploy_lambda.ps1 -LambdaIP <ip-address>
#
# This script:
#   1. Creates a tarball of the project (excluding data/results, __pycache__)
#   2. Uploads it to your Lambda Labs instance via scp
#   3. Extracts and runs the benchmark suite

param(
    [Parameter(Mandatory=$true)]
    [string]$LambdaIP,
    
    [string]$LambdaUser = "ubuntu",
    
    [int]$Duration = 10,

    [string]$SSHKey = ""
)

$ProjectDir = "E:\data\code\claudecode\TOMLSignals"
$TarName = "tomlsignals.tar.gz"
$RemoteDir = "~/TOMLSignals"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TOMLSignals - Lambda Labs Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Step 1: Package
Write-Host "`n[1/4] Packaging project..." -ForegroundColor Yellow
Push-Location $ProjectDir\..

# Use tar (available in Windows 10+)
tar -czf $TarName `
    --exclude='TOMLSignals/data/results/*' `
    --exclude='TOMLSignals/__pycache__' `
    --exclude='TOMLSignals/*/__pycache__' `
    --exclude='TOMLSignals/test_thermal.py' `
    TOMLSignals/

Write-Host "  Created $TarName ($('{0:N1}' -f ((Get-Item $TarName).Length / 1KB)) KB)"
Pop-Location

# Build SSH args
$SSHArgs = @()
if ($SSHKey -ne "") {
    $SSHArgs += @("-i", $SSHKey)
}

# Step 2: Upload
Write-Host "`n[2/4] Uploading to ${LambdaUser}@${LambdaIP}..." -ForegroundColor Yellow
scp @SSHArgs "$ProjectDir\..\$TarName" "${LambdaUser}@${LambdaIP}:~/"

# Step 3: Extract and setup
Write-Host "`n[3/4] Extracting on remote..." -ForegroundColor Yellow
ssh @SSHArgs "${LambdaUser}@${LambdaIP}" "cd ~ && tar -xzf $TarName && rm $TarName"

# Step 4: Run
Write-Host "`n[4/4] Starting benchmark suite (duration=${Duration}s)..." -ForegroundColor Yellow
Write-Host "  This will take 60-90 minutes. You can disconnect (uses nohup)." -ForegroundColor Gray
Write-Host ""

ssh @SSHArgs "${LambdaUser}@${LambdaIP}" @"
cd $RemoteDir && \
nohup bash -c 'bash run_lambda.sh --duration $Duration > benchmark_log.txt 2>&1' &
echo 'Benchmark started in background (PID: `$!`)'
echo 'Monitor: ssh ${LambdaUser}@${LambdaIP} tail -f ${RemoteDir}/benchmark_log.txt'
echo 'Results: scp -r ${LambdaUser}@${LambdaIP}:${RemoteDir}/data/results/ ./lambda_results/'
"@

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Deployment complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Monitor progress:" -ForegroundColor White
Write-Host "    ssh ${LambdaUser}@${LambdaIP} tail -f ${RemoteDir}/benchmark_log.txt"
Write-Host ""
Write-Host "  Download results when done:" -ForegroundColor White
Write-Host "    scp -r ${LambdaUser}@${LambdaIP}:${RemoteDir}/data/results/ .\lambda_results\"
Write-Host ""

# Clean up local tarball
Remove-Item "$ProjectDir\..\$TarName" -ErrorAction SilentlyContinue
