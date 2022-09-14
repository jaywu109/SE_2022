# Speech Enhancement 2022

Training script for speech enhacement challenge using [DB-AIAT](https://arxiv.org/abs/2110.06467) and [MetricGAN](https://arxiv.org/abs/1905.04874)

Environment and Requirement
---
1. Create environment using:
```
cd docker
docker build -t se_image .
docker run -itd --runtime nvidia --gpus all --name se_env -p se_image 
cd ..
```
2. Install additional packages:
```
pip install - r requirements.txt
```

Training and Testing
---
- Training DB-AIAT using DDP: 
```
cd model_experiment/db-aiat/training_script
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node 3 ddp_training.py 
```
- Run prediction using MetricGAN with `model_experiment/metricgan/metricgan.ipynb`


