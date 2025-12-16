# EndoStreamDepth
[EndoStreamDepth](https://openreview.net/pdf?id=I7lgdDdcij) has been submitted to MIDL2026.

## TODO
update readme with more details

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

## Citing EndoStreamDepth
If you find our EndoStreamDepth helpful, please use the following BibTeX entry.

```
@inproceedings{li2025endostreamdepth,
  title={EndoStreamDepth: Temporally Consistent Monocular Depth Estimation for Endoscopic Video Streams,
  author={Li, Hao and Lu, Daiwei, Wang, Jiacheng and Webster, Robert and Oguz, Ipek},
  booktitle={Medical Imaging in Deep Learning},
  year={2026 (submitted)}
}
```

## Contact
Email: hao.li.1@vanderbilt.edu

## Acknowledgements
[Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main), [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth/tree/main)
