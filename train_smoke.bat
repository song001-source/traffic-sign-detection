@echo off
title YOLO12 Training - COCO Traffic Signs

cd /d %~dp0

echo ============================================================
echo  YOLO12 Traffic Sign Training - COCO (46 classes)
echo ============================================================
echo.
echo  Model: yolo12n.pt
echo  Dataset: data/coco (46 classes, 6809 train + 1953 val)
echo  Epochs: 3 (smoke test)
echo.

python src\train.py --smoke

echo.
if %errorlevel% equ 0 (
    echo ============ TRAINING COMPLETE ============
) else (
    echo Training stopped or error occurred.
)
echo.
pause
