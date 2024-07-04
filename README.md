# Install-Walkthroughs IJW (It Just Works!)
## _Three Studio_
### ✨Python & PyTorch✨
```sh
conda create -n threestudio python==3.8.16
```
- Python==3.8.16
```sh
conda install pip==23.0.1
```
- Pip=23.0.1
```sh
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
- torch==2.3.0+cu121
- torchaudio==2.3.0+cu121
- torchvision==0.18.0+cu121
#
```sh
pip install lightning==2.3.1 omegaconf==2.3.0 jaxtyping==0.2.19 typeguard==4.3.0 nerfacc==0.5.3 tinycudann@git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch diffusers==0.19.3 transformers==4.28.1 accelerate==0.32.0 opencv-python==4.10.0.84 tensorboard==2.14.0 matplotlib==3.7.5 imageio==2.34.2 imageio-ffmpeg==0.5.1 nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@c5caf7bdb8a2448acc491a9faa47753972edd380 libigl==2.5.1 xatlas==0.0.9 trimesh==4.4.1 networkx==3.0 pysdf==0.1.9 PyMCubes==0.1.4 wandb==0.17.4 gradio==4.11.0 envlight@git+https://github.com/ashawkey/envlight.git@05b5851e854429d72ecaf5b206ed64ce55fae677 torchmetrics==1.4.0.post0 xformers==0.0.26.post1 bitsandbytes==0.38.1 sentencepiece==0.2.0 safetensors==0.4.3 huggingface-hub==0.23.4 einops==0.8.0 kornia==0.7.3 taming-transformers-rom1504==0.0.6 clip@git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1 controlnet-aux==0.0.9
```
- lightning==2.3.1
- omegaconf==2.3.0
- jaxtyping==0.2.19
- typeguard==4.3.0
- nerfacc==0.5.3
- tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
- diffusers==0.19.3
- transformers==4.28.1
- accelerate==0.32.0
- opencv-python==4.10.0.84
- tensorboard==2.14.0
- matplotlib==3.7.5
- imageio==2.34.2
- imageio-ffmpeg==0.5.1
- nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@c5caf7bdb8a2448acc491a9faa47753972edd380
- libigl==2.5.1
- xatlas==0.0.9
- trimesh==4.4.1
- networkx==3.0
- pysdf==0.1.9
- PyMCubes==0.1.4
- wandb==0.17.4
- gradio==4.11.0
- envlight @ git+https://github.com/ashawkey/envlight.git@05b5851e854429d72ecaf5b206ed64ce55fae677
- torchmetrics==1.4.0.post0
- torchmetrics==1.4.0.post0
#
#### deepfloyd
- xformers==0.0.26.post1
- bitsandbytes==0.38.1
- sentencepiece==0.2.0
- safetensors==0.4.3
- huggingface-hub==0.23.4
#
#### zero123
- einops==0.8.0
- kornia==0.7.3
- taming-transformers-rom1504==0.0.6
- clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
#
#### controlnet
- controlnet-aux==0.0.9






transformers==4.28.1
trimesh==4.4.1
triton==2.3.0
wandb==0.17.4
