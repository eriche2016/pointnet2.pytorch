#!/bin/bash

# download modelnet40 pcl data 
if [ 1 -eq 0 ]; then
    python ./download_scripts/download.py --dataset modelnet40_pcl
fi

# run classification experiments on modelnet40 point cloud 
if [ 1 -eq 0 ]; then 
    python main_cls.py --cuda --gpu_id 3 
fi

if [ 1 -eq 0 ]; then 
    python main_cls.py --cuda \
    --gpu_id 3 \
    --init_model ./models_checkpoint/model_best.pth \
    --optim_state_from ./models_checkpoint/optim_state_best.pth
fi 

# download original seg data 
if [ 1 -eq 0 ]; then
    python ./download_scripts/download.py --dataset shapenetcore_partanno
fi
# dowload seg h5 data 
if [ 1 -eq 0 ]; then
    python ./download_scripts/download.py --dataset shapenetcore_partanno_h5
fi

if [ 1 -eq 0 ]; then
    python ./download_scripts/download.py --dataset shapenetcore_partanno_ben_v0
fi
###########################################
# run part segmentation experiments on shapenetcore_partanno
###########################################
if [ 1 -eq 1 ]; then 
    python main_part_seg.py --cuda --gpu_id 3 
fi
###########################################
# test part seg, note data from folder  
###########################################
if [ 1 -eq 0 ]; then 
    python ./eval_part_seg_folder.py
fi 
if [ 1 -eq 0 ]; then 
    python eval_part_seg_h5.py 
fi 


############################################
## visulization 
############################################
if [ 1 -eq 0 ]; then
    echo 'build cpp code for visualization of 3d point data'
    cd ./tools/visualizations/ 
    sh build.sh
    cd ..
    cd .. 
fi
if [ 1 -eq 0 ]; then
    python ./tools/visualizations/show3d_balls.py
fi 
