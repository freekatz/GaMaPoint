#!/usr/bin/env bash

set -x
OPENPOINTS_BRANCH=$1

PWD=$(cd "$(dirname "$0")"/../;pwd)
HACK_DIR=$PWD/hack
OPENPOINTS_DIR=$PWD/openpoints
cd $PWD

source activate
conda deactivate
conda activate openpoints

# setup packages
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html
pip install -r $HACK_DIR/requirements.txt
# https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp38-cp38-linux_x86_64.whl

# setup openpoint libs
#git submodule update --init --recursive
#git submodule update --remote --merge # update to the latest version
#cd $OPENPOINTS_DIR
#git switch ${OPENPOINTS_BRANCH}
#cd $PWD''

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
pip install .
cd ../
