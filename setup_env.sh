
# torch version must be under (including) 2.4
pip install torch==2.4.0 torchvision==0.19.0
pip install xformers==0.0.27.post2

pip install ipdb tqdm wandb
pip install matplotlib einops scipy h5py OpenEXR
pip install hydra-core
pip install opencv-python pillow

pip install flash-attn --no-build-isolation


# install our local mamba package
cd mamba
export MAMBA_FORCE_BUILD=TRUE
python -m pip install --no-build-isolation .
cd ..

sudo apt install -y ffmpeg
