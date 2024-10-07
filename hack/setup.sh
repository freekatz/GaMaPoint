#!/usr/bin/env bash

set -x
GAMA_BRANCH=$1

PWD=$(cd "$(dirname "$0")"/../;pwd)
HACK_DIR=$PWD/hack
cd $PWD

# activate env
source activate
conda deactivate
conda activate mtxai

# setup packages
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
# https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r $HACK_DIR/requirements.txt

# install 4d kdtree cuda extensions
cd $PWD/backbone/ops/pykdtree
pip install .
cd $PWD
