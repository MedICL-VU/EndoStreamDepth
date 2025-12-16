# EndoStreamDepth: Temporally Consistent Monocular Depth Estimation for Endoscopic Video Streams

[EndoStreamDepth](https://openreview.net/pdf?id=I7lgdDdcij) has been submitted to MIDL2026.


**EndoStreamDepth** is a real-time *video stream* monocular depth estimation framework for endoscopic videos. It predicts **accurate**, **temporally consistent**, and **sharp-boundary** metric depth maps by processing frames **sequentially (streaming)** with hierarchical temporal modules.

## TODO
update readme with more details


## Highlights

- **Streaming inference (frame-by-frame):** no batched multi-frame processing, low latency for real-world deployment.
- **Endoscopy-Specific Transformation (EST):** geometry + photometric perturbations tailored for endoscopy.
- **Hierarchical multi-level temporal modeling:** Mamba modules at multiple decoder scales to stabilize predictions.
- **Comprehensive supervision:** multi-scale supervision + metric + edge losses + self-supervised temporal regularization.



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
