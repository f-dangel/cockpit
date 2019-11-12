#!/bin/bash

# Create conda environment named 'backboard_env'

set -e
green='\e[1;32m%s\e[0m\n'

conda create --force --yes --name backboard_env python=3.7 pip
source ~/anaconda3/bin/activate backboard_env

# Dependencies, torch with CUDA
conda install --yes pytorch torchvision -c pytorch

# Additional DeepOBS requirements (from website)
pip install argparse matplotlib2tikz numpy pandas matplotlib seaborn bayesian-optimization

# Backpack, optimizers, integration with DeepOBS
pip install -r requirements.txt
# pip install -e .

printf "\n$green\n" '[Activate by] conda activate backpack_paper_env'

conda deactivate
