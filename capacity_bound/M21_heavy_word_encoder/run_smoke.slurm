#!/bin/bash
#SBATCH --job-name=m21_smoke
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M21_heavy_word_encoder"
python -u -c "
import torch, models as M
m=M.build(30).cuda()
print('params:', f'{m.count_params():,}')
x=torch.randn(4,300000,device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):
    out=m(x); loss=out.float().sum()
loss.backward()
print('forward+backward OK, logits', tuple(out.shape))
"
