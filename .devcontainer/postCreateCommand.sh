#!/usr/bin/env bash

# set -ex

echo "******************************************"
echo "Welcome to the PyTorch Docker from Jearde!"
echo "******************************************"

if [ -z "$GIT_NAME" ]
then
    echo "GIT_NAME is not defined. Not setting git config."
else 
    echo "Setting git config..."
    git config --global user.email $GIT_NAME
    git config --global user.name $GIT_EMAIL
    git config --global core.editor "code-insiders --wait"
    git config --global pull.rebase true
    git config --global push.default current
    git config --global --add --bool push.autoSetupRemote true
fi

echo "Container specifications:"
echo ""

echo "User: $(whoami)"
echo "OS: $(lsb_release -a | grep Description:)"
echo "Python: $(which python)"
echo "Python Version: $(python -V)"
nvidia-smi
nvidia-smi -L
python -c '
import torch
import torchvision
import torchaudio
import lightning as L
print(f"PyTorch Version: {torch.__version__}")
print(f"Lightning Version: {L.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
'
echo "*******************************************************************************"