#!/bin/bash
export CAR_PATH=/home/gzh/anycar
export LD_LIBRARY_PATH=/home/gzh/miniconda3/envs/anycar/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/gzh/miniconda3/envs/anycar/lib/python3.10/site-packages:/home/gzh/anycar/car_dynamics/car_dynamics

source install/setup.bash
cp -r  /home/gzh/anycar/car_foundation/car_foundation/models  /home/gzh/anycar/install/car_foundation/lib/python3.10/site-packages/car_foundation/models
cp -r /home/gzh/anycar/car_planner/assets /home/gzh/anycar/install/car_planner/lib/python3.10/site-packages/

export PATH=$PATH:/home/gzh/miniconda3/bin

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
