#!/bin/bash
#SBATCH --job-name=m21_smoke2
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=smoke2_%j.out
#SBATCH --error=smoke2_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M21_heavy_word_encoder"
python -u -c "
import torch, time, models as M
m=M.build(30).cuda()
x=torch.randn(12,300000,device='cuda')
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
import torch.nn.functional as F
for i in range(6):
    t=time.time()
    with torch.autocast('cuda',dtype=torch.float16):
        loss=F.cross_entropy(m(x),torch.zeros(12,dtype=torch.long,device='cuda'))
    loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); print(f'step {i}: {time.time()-t:.2f}s')
"
