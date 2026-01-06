# EndoStreamDepth: Temporally Consistent Monocular Depth Estimation for Endoscopic Video Streams

[![Paper](https://img.shields.io/badge/paper-arXiv-green)](https://arxiv.org/abs/2512.18159)

This work has been submitted to MIDL2026, here is the [paper](https://arxiv.org/abs/2512.18159)


**EndoStreamDepth** is a real-time *video stream* monocular depth estimation framework for endoscopic videos, running at **24 FPS** at 518×518 resolution. It predicts **accurate**, **temporally consistent**, and **sharp-boundary** metric depth maps by processing frames **sequentially (streaming)** with hierarchical temporal modules.


## Highlights

- **Streaming inference (frame-by-frame):** no batched multi-frame processing, low latency for real-world deployment.
- **Endoscopy-Specific Transformation (EST):** geometry + photometric perturbations tailored for endoscopy.
- **Hierarchical multi-level temporal modeling:** Mamba modules at multiple decoder scales to stabilize predictions.
- **Comprehensive supervision:** multi-scale supervision + metric + edge losses + self-supervised temporal regularization.


##
![EndoStreamDepth](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/figs/EndoStreamDepth_overview_horizontal.png?raw=true)
We provide both a single-frame baseline (left-b) and a streaming video model (left-c) with multiple cupervision signials. The streaming model further adds multi-level temporal modules (right) to propagate information over time for stable metric depth.


## Results at a glance
- **Public datasets:** evaluated on public dataset **C3VD** and **SimCol3D**.
- **Strong performance:** achieves **state-of-the-art** accuracy and boundary quality against strong baselines (e.g., DepthAnything v2, FlashDepth, Metric DAv2).
- **Benchmark coverage:** validated under **in-distribution**, **domain-shift**, and **cross-dataset** settings to demonstrate robustness.
- **Temporal stability:** improves frame-to-frame consistency (lower flicker) while keeping **streaming** inference.


Details can be viewed in our [paper](https://openreview.net/pdf?id=I7lgdDdcij).

## 

![Qualitative](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/figs/qualitative.png?raw=true)
Compared to FlashDepth and Metric Depth Anything v2, EndoStreamDepth produces lower error maps, a cleaner depth shape, and sharper depth discontinuities around anatomical boundaries, with fewer local artifacts highlighted by the arrows.


## Installation
```
git clone https://github.com/MedICL-VU/EndoStreamDepth.git
cd EndoStreamDepth
conda create -n endostreamdepth python=3.11 --yes
conda activate endostreamdepth
bash setup_env.sh
```


## Training
 ```
/path/to/conda/env/bin/torchrun
--nproc_per_node=1 \
train.py \
--config-path configs/endostreamdepth/<your_config>.yaml \
dataset.data_root=<your_dataset_root>
```

 
or similiar to ```/home/xx/anaconda3/envs/xxx/bin/torchrun```, which is your conda env path


## Inference
```
/path/to/conda/env/bin/torchrun
--nproc_per_node=1 \
train.py \
--config-path configs/endostreamdepth/<your_config>.yaml \
inference=true \
dataset.data_root=<your_dataset_root> \
load=/path/to/your/weights.pth
```

### Inference example ###

If you are using the example [dataset](https://github.com/MedICL-VU/EndoStreamDepth/tree/main/cv3d/trans_t1_a), then you can change the test video [here](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/dataloaders/cv3d_dataset.py#L10), leaving first video along, and run below command

```
torchrun --nproc_per_node=1 \
      train.py \
      --config-path configs/example_inference \
      inference=true
```

## Pretrained Weights
| C3VD (split1) | C3VD (split2) | SimCol3D |
|------------------------------|------------------------------|------------------------------|
|[Download](https://drive.google.com/drive/u/1/folders/1XKaeCeuhaDlO8F8knfPlyjLTaaPxoDLE)| [Download](https://drive.google.com/drive/u/1/folders/1J1vsEz30J0SVpMt2YB9AOqff4Wjpz5kN) |[Download](https://drive.google.com/drive/u/1/folders/14byiSecftSH116Kcdu8z55nOBegxNQgJ)|

update the pretrained weights path [here](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/configs/endostreamdepth/config.yaml#L4) or other custom config files.

## Datasets and Splits

### C3VD (Phantom colonoscopy depth dataset)
- 22 video sequences, 10,015 frames with paired ground-truth depth.
- Four colon segments: cecum, descending colon, sigmoid colon, transverse colon.
- We use two evaluation splits:
  - **Split 1 (domain shift):** hold out transverse colon videos for testing
  - **Split 2 (in-distribution)**

### SimCol3D (Simulated colonoscopy depth dataset)
- 33 videos, 37,800 frames with paired depths.
- Follow the official training/evaluation splits.

### Data split
- [C3VD split 1](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/dataloaders/cv3d_dataset.py#L10)
- [C3VD split 2](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/dataloaders/cv3d2_dataset.py#L10)
- [SimCol3D](https://github.com/MedICL-VU/EndoStreamDepth/blob/main/dataloaders/simcol3d.py#L10)
## Quantitative Results

### C3VD (Split 1, in-distribution):transverse colon is held out for testing.

**EndoStreamDepth** achieves the best performance across all metrics, including both global geometry (δ1/AbsRel/RMSE/L1) and boundary sharpness (F1).

🏆 best

| Method | δ1 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE log ↓ | L1 ↓ | F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DepthAnything v2 | 0.847 | 0.158 | 0.635 | 3.497 | 0.169 | 2.503 | 0.089 |
| Metric DAv2 | 0.850 | 0.149 | 0.566 | 3.538 | 0.166 | 2.492 | 0.095 |
| EndoOmni | 0.836 | 0.154 | 0.610 | 3.623 | 0.170 | 2.596 | 0.109 |
| DINOv3 depth | 0.731 | 0.192 | 1.188 | 5.457 | 0.194 | 3.955 | 0.070 |
| FlashDepth | 0.730 | 0.188 | 1.046 | 4.989 | 0.190 | 3.780 | 0.116 |
| EndoStreamDepth (Ours) | 0.952 🏆 | 0.085 🏆 | 0.246 🏆 | 2.739 🏆 | 0.107 🏆 | 1.780 🏆 | 0.143 🏆 |


### Benchmarking: in-distribution + cross-dataset generalization

🏆 best

**C3VD (Split 2, in-distribution)**

| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ |
|---|---:|---:|---:|
| LightDepth | 0.078 | 1.81 | 6.55 |
| NormDepth+ | 0.155 | 1.53 | 7.51 |
| PPSNet-Teacher | 0.053 | 0.15 | 2.15 |
| PPSNet-Student | 0.049 🏆 | 0.14 | 2.06 |
| Ours-frame | 0.077 | 0.27 | 1.74 |
| Ours-video | 0.052 | 0.11 🏆 | 1.72 🏆 |


**SimCol3D (SimCol III test, unseen sequence)**

🏆 best

| Method | L1 ↓ | RMSE ↓ | AbsRel ↓ |
|---|---:|---:|---:|
| CVML (1st) | 0.099 | 0.141 | 0.025 |
| MIVA (2nd) | 0.107 | 0.163 | 0.025 |
| EndoAI (3rd) | 0.111 | 0.168 | 0.028 |
| IntuitiveIL (4th) | 0.167 | 0.233 | 0.047 |
| Ours-frame | 0.099 | 0.140 | 0.028 |
| Ours-video | 0.087 🏆 | 0.126 🏆 | 0.023 🏆 |



## Citing EndoStreamDepth
If you find our EndoStreamDepth helpful, please use the following BibTeX entry.

```bibtex
@article{li2025endostreamdepth,
  title={EndoStreamDepth: Temporally Consistent Monocular Depth Estimation for Endoscopic Video Streams},
  author={Li, Hao and Lu, Daiwei and Wang, Jiacheng and Webster III, Robert J and Oguz, Ipek},
  journal={arXiv preprint arXiv:2512.18159},
  year={2025}
}
```

## Contact
Email: hao.li.1@vanderbilt.edu

## Acknowledgements
[Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main), [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth/tree/main)
