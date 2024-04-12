# 应用环境搭建

## 一、基础软件安装

### 1、linux

#### （1）推理卡驱动安装

```shell
# 1、更新操作系统
# ubuntu 只需要一步
sudo apt update
# rhel 需要更新开发包
sudo yum update
# 根据uname -r查出kernel版本号
uname -r
# 比如版本号为3.10.0-957.el7.x86_64，则执行以下命令
sudo yum install gcc kernel-devel-3.10.0-957.el7.x86_64 kernel-headers
# 2、安装编译环境
# ubuntu
sudo apt install g++
sudo apt install gcc
sudo apt install make
# rhel
sudo yum install g++
sudo yum install gcc
sudo yum install make
sudo yum install wget
# 3、禁用 nouveau
# 编辑blacklist.conf文件
sudo vi /etc/modprobe.d/blacklist.conf
# 结尾处增加以下两行
blacklist nouveau
options nouveau modeset=0
# 保存后退出
# 对上面修改的文件进行更新
# ubuntu
sudo apt install dracut
sudo dracut --force
# rhel
sudo update-initramfs -u
# 重启系统
sudo reboot
# 验证是否禁用了nouveau，显示为空说明成功禁用
lsmod | grep nouveau 
# 4、关闭显示服务
sudo telinit 3
sudo service gdm3 stop
# 5、下载驱动
wget https://us.download.nvidia.cn/XFree86/Linux-x86_64/525.85.05/NVIDIA-Linux-x86_64-525.85.05.run
# 6、安装驱动
chmod +x NVIDIA-Linux-x86_64-525.85.05.run
sudo ./NVIDIA-Linux-x86_64-525.85.05.run
# 安装过程中有一个warning或选择项，选默认项即可
# 7、验证
nvidia-smi
```

#### （2）CUDA安装

```shell
#　1、下载
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
# 2、安装
chmod +x cuda_12.0.1_525.85.12_linux.run 
sudo ./cuda_12.0.1_525.85.12_linux.run
# 3、增加环境变量
vi ~/.bashrc
# 增加以下两行
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64
# 环境变量生效
source ~/.bashrc
# 4、验证
nvcc -V
```

#### （3）Anaconda安装

```shell
# 1、下载
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# 2、安装
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh
# 3、环境变量生效
source ~/.bashrc
# 4、验证
conda -V
```

#### （4）验证开发环境

```shell
# 1、创建python虚拟环境
conda create -n test python=3.10 -y
conda activate test
# 2、安装PyTorch
pip install torch==2.0.1 -i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 3、验证PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

#### （5）PyTorch重装

```shell
# 1、卸载
pip uninstall torch -y 
pip uninstall torchvision -y
# 2、安装旧版本torch
# 如CUDA 12.0
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
--index-url https://download.pytorch.org/whl/cu121
# 如CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 3、验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 2、windows

```shell
# 1、CUDA
https://developer.nvidia.com/cuda-toolkit-archive
# 2、Anaconda
https://www.anaconda.com/
```

## 二、其他软件安装

### 1、nginx

#### （1）ubuntu

```shell
# 1、安装依赖
sudo apt install curl gnupg2 ca-certificates lsb-release ubuntu-keyring
# 2、导入官方 nginx 签名密钥
curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor \
    | sudo tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null
# 3、为稳定版 nginx 软件包设置 apt 存储库
echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] \
http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" \
    | sudo tee /etc/apt/sources.list.d/nginx.list
# 4、更新
sudo apt update
# 5、安装指定版本（ubuntu22.04）
sudo apt install nginx=1.24.0-1~jammy
# 注意，如果是ubuntu20.04，则使用
sudo apt install nginx=1.24.0-1~focal
```

#### （2）Redhat

```shell
sudo yum install nginx 
```

#### （3）Windows

```shell
https://nginx.org/en/download.html
```

### 2、git

#### （1）linux

```shell
# ubuntu
sudo apt update
sudo apt install git
# rhel
sudo yum update
sudo yum install git
# 验证
git --version
# 此命令显示git的版本号
```

#### （2）Windows

```shell
https://git-scm.com/download/win
```

