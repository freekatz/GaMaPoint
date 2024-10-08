#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J seg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G

[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs
#
#module load cuda/11.1.1
#module load gcc

echo "===> Anaconda env loaded"
source activate openpoints

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo $NUM_GPU_AVAILABLE


source=$1
cfg=$2
PY_ARGS=${@:3}
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU_AVAILABLE} $source --cfg $cfg --gpus ${NUM_GPU_AVAILABLE} ${PY_ARGS}


# how to run
# using slurm, run with 1 GPU, by 3 times (array=0-2):
# sbatch --array=0-2 --gres=gpu:1 --time=12:00:00 script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# if using local machine with GPUs, run with ALL GPUs:
# bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# local machine, run with 1 GPU:
# CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml
