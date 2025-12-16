# EndoStreamDepth
EndoStreamDepth
has been submitted to MIDL2026. I'm cleaning the code and will upload it. Contact info: hao.li.1@vanderbilt.edu




## Installation
```
conda create -n flashdepth python=3.11 --yes
conda activate flashdepth
bash setup_env.sh
```


## Training
 ```/home/xx/anaconda3/envs/xxx/bin/torchrun --nproc_per_node=1 train.py --config-path configs/endostreamdepth/ load=checkpoints/your_checkpoint dataset.data_root=your_path```

 
```/home/xx/anaconda3/envs/xxx/bin/torchrun``` is your conda env path


## Inference
``` /home/xx/anaconda3/envs/xxx/bin/torchrun   --nproc_per_node=1   train.py   --config-path configs/endostreamdepth/   inference=true   dataset.data_root=your_path```
