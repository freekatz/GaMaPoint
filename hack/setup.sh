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

# install pybind
sudo apt update
sudo apt-get install python3-pybind11

# setup python packages
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
# wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt212.tar.bz2
#conda install pytorch3d-0.7.8-py312_cu121_pyt231.tar.bz2
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r $HACK_DIR/requirements.txt

# setup kdtree cuda extensions
sudo apt-get install libomp-dev
cd $PWD/backbone/ops/pykdtree
pip install .
cd $PWD


# install eigen (if setup pykdtree failed)
sudo apt-get install libeigen3-dev
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

# setup openmp (if setup pykdtree failed)
sudo apt-get install cmake autoconf automake libtool flex
autoreconf -f -i
mkdir $PWD/tmp
cd $PWD/tmp
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -zxf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
./configure --prefix=$PWD/tmp/openmpi-4.1.6
make -j8
sudo make install
cd $PWD
sudo mv $PWD/tmp/openmpi-4.1.6 /usr/local/openmpi
export PATH="$PATH:/usr/local/openmpi/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/openmpi/lib"
