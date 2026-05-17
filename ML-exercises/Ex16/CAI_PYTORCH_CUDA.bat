@echo off
chcp 65001 >nul
echo ============================================
echo  CAI PyTorch CO CUDA (Anaconda)
echo ============================================
echo.

set PY=C:\Users\ADMIN\anaconda3\python.exe
if not exist "%PY%" (
    echo Khong tim thay: %PY%
    echo Sua bien PY trong file .bat neu Anaconda o cho khac.
    pause
    exit /b 1
)

echo [1/4] Kiem tra NVIDIA...
nvidia-smi
if errorlevel 1 (
    echo.
    echo LOI: nvidia-smi khong chay duoc. Cap nhat driver NVIDIA truoc.
    pause
    exit /b 1
)

echo.
echo [2/4] Go PyTorch cu...
"%PY%" -m pip uninstall -y torch torchvision torchaudio 2>nul

echo.
echo [3/4] Cai PyTorch CUDA 12.4 (co the mat 5-15 phut)...
"%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo Thu lai voi CUDA 12.1...
    "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 (
    echo Thu lai voi CUDA 11.8...
    "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [4/4] Kiem tra...
"%PY%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built:', torch.backends.cuda.is_built()); print('CUDA available:', torch.cuda.is_available());
import sys
sys.exit(0 if torch.cuda.is_available() else 1)"

if errorlevel 1 (
    echo.
    echo VAN CHUA CO CUDA. Thu conda trong Anaconda Prompt:
    echo   conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
    pause
    exit /b 1
)

echo.
echo ============================================
echo  THANH CONG! Mo Jupyter - Restart Kernel
echo ============================================
pause
