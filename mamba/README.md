# Mamba installation

Code is taken from https://github.com/state-spaces/mamba

We've made some changes to the file `modules/mamba2.py`. Feel free to check the comments in the file for details. 
Install the local files using the following commands (or simply run our `setup_env.sh` script):
```
cd $this_directory
export MAMBA_FORCE_BUILD=TRUE
python -m pip install --no-build-isolation .
```

## Citation

All credit of Mamba goes to the original authors. If you use our repo, please also cite:
```
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

```
