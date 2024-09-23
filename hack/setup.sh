#!/usr/bin/env bash

set -x
GAMA_BRANCH=$1

PWD=$(cd "$(dirname "$0")"/../;pwd)
HACK_DIR=$PWD/hack
cd $PWD

# activate env
source activate
conda deactivate
conda activate gama_points

# setup packages
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html
pip install -r $HACK_DIR/requirements.txt

# install kdtree cuda extensions
cd $PWD/utils/3rd/torch_kdtree
pip install .
cd $PWD
