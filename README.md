# Ubuntu 🔥

## CUDA! CUDA! CUDA!
### Remove Nvidia drivers & CUDA & CuDNN Libraries
```sh
sudo apt-get --purge -y remove 'cuda*'
sudo apt-get --purge -y remove 'nvidia*'
sudo apt-get autoremove --purge cuda
sudo rm -rf /usr/local/cuda*
sudo apt-get --purge remove '*cud*'
sudo apt-get autoremove --purge '*cud*'


find / -name '*cuda*'
sudo dpkg -l | grep nvidia
sudo dpkg -l | grep cuda
sudo apt-get remove --purge <Package Names>
sudo apt-get autoclean
```
For edit of path
```sh
nano ~/.bashrc
source ~/.bashrc
```
Disable nouveau
```sh
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
```
If it shows the following, you have done it correctly
```sh
blacklist nouveau
options nouveau modeset=0
```
Update Kernel
```sh
sudo update-initramfs -u
```
#
### Installing Nvidia Drivers
Check current GPU model
```sh
lshw -numeric -C display
lspci | grep -i nvidia
```
Check what drivers are available
```sh
ubuntu-drivers devices
```
1.Install the recommended driver
```sh
sudo ubuntu-drivers autoinstall
```
2.Install a certain version
```sh
sudo apt install nvidia-driver-<NUMBER>
```
### Installing CUDA Toolkit
| CUDA | Link |
| ------ | ------ |
| 12.1 | [developer.nvidia.com/cuda-12-1-0-download-archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)|
```sh
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
```
```sh
sudo sh cuda_12.1.0_530.30.02_linux.run
```
🚨 Only Check the CUDA Toolkit and do not check the drivers, as they are outdated and fail to install in the first place

### Installing cuDNN
| cuDNN | Link |
| ------ | ------ |
| 9.2.1 | [developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)|
```sh
wget https://developer.download.nvidia.com/compute/cudnn/9.2.1/local_installers/cudnn-local-repo-ubuntu2004-9.2.1_1.0-1_amd64.deb
```
```sh
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.2.1_1.0-1_amd64.deb
```
```sh
sudo cp /var/cudnn-local-repo-ubuntu2004-9.2.1/cudnn-*-keyring.gpg /usr/share/keyrings/
```
```sh
sudo apt-get update
```
```sh
sudo apt-get -y install cudnn-cuda-12
```
#
### Reset Ubuntu 💥

```sh
sudo dpkg --configure -a
sudo apt-get update
sudo apt-get -f install
sudo apt-get full-upgrade
sudo apt-get install --reinstall ubuntu-desktop
sudo apt-get autoremove
sudo apt-get clean
```
## Installing latest version of CMake
![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlxstUPxXgvjpaU5uOf7R-HjTgVtjow2EyTw&s)

[Download CMake-<>.tar.gz](https://cmake.org/download/)
```sh
wget link
tar zxf cmake-<>.tar.gz
cd cmake-<>
./bootstrap
make
sudo make install
```
## Installing Xrdp & Xubuntu-desktop
Install Fortivpnclient
https://www.fortinet.com/support/product-downloads

Installing Xubuntu Desktop
```sh
sudo apt update
sudo apt install xubuntu-desktop
```
Installing Xrdp
```sh
sudo apt update
sudo apt install xrdp
sudo systemctl status xrdp
```
Adding SSL access authentication to Xrdp
```sh
sudo adduser xrdp ssl-cert
sudo systemctl restart xrdp
```
(Needed for certain people)Setting Fortivpnclient to attach to certain IP

Auto permission for color etc in Xrdp
```sh
sudo mkdir -p /etc/polkit-1/localauthority/50-local.d
sudo nano /etc/polkit-1/localauthority/50-local.d/color.pkla
```
Edit as following
```sh
[Allow the actions for xRDP]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.NetworkManager.wifi.scan
ResultAny=yes
ResultInactive=yes
ResultActive=yes
```

# Tips & Commands
Environment variables
```sh
nano ~/.bashrc
source ~/.bashrc
```
Login as Administrator
```sh
sudo -i
```
## Shortcuts
Open Terminal
```sh
crtl+alt+t
```
Save and Exit in Nano
```sh
crtl+s
crtl+x
```
#
# Install-Walkthroughs IJW (It Just Works!)
## _Stable Diffusion WebUI with conda_
### ✨Python & PyTorch✨
```sh
conda create -n MainTP python==3.10.6
```
- Python==3.10.6

### Set main python path to that conda environment
```sh
nano ~/.bashrc
add this line -> export PATH=$HOME/<anaconda3 or miniconda3>/envs/MainTP/bin:$PATH
source ~/.bashrc
```
Install libg
```sh
sudo apt install libgl1 libglib2.0-0
```
Navigate to the directory for Stable-diffusion-webui
```sh
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
```
Run webui.sh with xformers
```sh
bash webui.sh --xformers
```
Check webui-user.sh for options.

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
pip install ninja==1.11.1.1 lightning==2.3.1 omegaconf==2.3.0 jaxtyping==0.2.19 typeguard==4.3.0 nerfacc==0.5.3 tinycudann@git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch diffusers==0.19.3 transformers==4.28.1 accelerate==0.32.0 opencv-python==4.10.0.84 tensorboard==2.14.0 matplotlib==3.7.5 imageio==2.34.2 imageio-ffmpeg==0.5.1 nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@c5caf7bdb8a2448acc491a9faa47753972edd380 libigl==2.5.1 xatlas==0.0.9 trimesh==4.4.1 networkx==3.0 pysdf==0.1.9 PyMCubes==0.1.4 wandb==0.17.4 gradio==4.11.0 envlight@git+https://github.com/ashawkey/envlight.git@05b5851e854429d72ecaf5b206ed64ce55fae677 torchmetrics==1.4.0.post0 xformers==0.0.26.post1 bitsandbytes==0.38.1 sentencepiece==0.2.0 safetensors==0.4.3 huggingface-hub==0.23.4 einops==0.8.0 kornia==0.7.3 taming-transformers-rom1504==0.0.6 clip@git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1 controlnet-aux==0.0.9
```
- ninja==1.11.1.1
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

Clone the repository
```sh
git clone https://github.com/threestudio-project/threestudio
```
