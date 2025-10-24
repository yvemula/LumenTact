# ================================
# SANPO selective downloader (PowerShell version, fixed regex escaping)
# ================================

$ErrorActionPreference = "Stop"

$SRC = "gs://gresearch/sanpo_dataset/v0/sanpo-real"
$DEST = "C:\Users\jc219\OneDrive\Desktop\GitHub\LumenTact\sanpo_snippet"

# Properly escape regex for PowerShell (single quotes prevent interpretation)
$EXCLUDE_REGEX = '(.*/camera_right/.*|.*/imu/.*|.*/depth_zed/.*)'

$sessions = @(
    "rh8DI_ycaCHwNQv3qxmpSYDYJX_78TiT",
    "sgb9jS9iEBepAPRt-Ijrx83GuV2JPat-",
    "skUAOmfq6qSwdCi3W6k2w6XCfS16GpP7",
    "tGlycnI_T0VHy6jKFKcI728lWqMd7d21",
    "uKY31hW60Rf2xoklHN7PICIryxWkhsM0"
)

New-Item -ItemType Directory -Force -Path $DEST | Out-Null

foreach ($id in $sessions) {
    Write-Host "===============================" -ForegroundColor Cyan
    Write-Host ("Downloading session: " + $id) -ForegroundColor Green
    Write-Host "===============================" -ForegroundColor Cyan

    $target = Join-Path $DEST $id
    New-Item -ItemType Directory -Force -Path $target | Out-Null

    $srcPath = "$SRC/$id/"
    $destPath = "$target\"

    Write-Host ("Syncing from " + $srcPath + " to " + $destPath + "...") -ForegroundColor Yellow

    # Explicitly call gsutil with array-form arguments (no shell parsing)
    & gsutil -m rsync -r $srcPath $destPath

    Write-Host ("Finished session: " + $id) -ForegroundColor Green
    Write-Host ""
}

Write-Host "All selected sessions downloaded successfully!" -ForegroundColor Cyan
