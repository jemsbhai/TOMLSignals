# NCU scaling experiments for SVD and JPEG
# Run from E:\data\code\claudecode\TOMLSignals

$metrics = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum"

$outdir = "data\ncu_profiles\scaling"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

Write-Host "=== SVD: Vary N, fixed D=64 ===" -ForegroundColor Cyan
foreach ($N in @(256, 512, 1024, 2048, 4096)) {
    $outfile = "$outdir\svd_N${N}_D64.csv"
    Write-Host "  SVD N=$N D=64 -> $outfile"
    ncu --metrics $metrics --csv --profile-from-start off python profile_scaling.py --alg svd --N $N --D 64 2>$null | Out-File $outfile -Encoding utf8
}

Write-Host "`n=== SVD: Vary D, fixed N=1024 ===" -ForegroundColor Cyan
foreach ($D in @(16, 32, 64, 128)) {
    $outfile = "$outdir\svd_N1024_D${D}.csv"
    Write-Host "  SVD N=1024 D=$D -> $outfile"
    ncu --metrics $metrics --csv --profile-from-start off python profile_scaling.py --alg svd --N 1024 --D $D 2>$null | Out-File $outfile -Encoding utf8
}

Write-Host "`n=== JPEG: Vary image size ===" -ForegroundColor Cyan
foreach ($N in @(256, 1024, 4096, 16384)) {
    $side = [math]::Floor([math]::Sqrt($N))
    $side = $side - ($side % 8)
    $nblocks = [math]::Pow($side / 8, 2)
    $outfile = "$outdir\jpeg_N${N}_side${side}.csv"
    Write-Host "  JPEG N=$N side=$side blocks=$nblocks -> $outfile"
    ncu --metrics $metrics --csv --profile-from-start off python profile_scaling.py --alg jpeg --N $N 2>$null | Out-File $outfile -Encoding utf8
}

Write-Host "`nDone! CSV files in $outdir" -ForegroundColor Green
Write-Host "Run: python parse_scaling_ncu.py  to analyze results"
