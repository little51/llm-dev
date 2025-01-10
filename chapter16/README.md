# 第16章 数字人应用

## 一、运行环境要求

```shell
OS    ：windows或linux
GPU   ：显存6G以上
CUDA  ：12.0以上
Torch ：2.1.2及以上
```

## 二、Python虚拟环境建立

```shell
# 建立python3.10环境
conda create -n meta-human python=3.10 -y
conda activate meta-human
# 更新 pip
python -m pip install --upgrade pip
```

## 三、SadTalker安装

### 1、代码下载

```bash
# clone 代码
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
git checkout cd4c046
```


### 2、安装基础依赖库

```bash
# 使用conda安装ffmpeg，注意不是pip，因为ffmpeg是在操作系统上调用，不是在python环境里
conda install ffmpeg -y
# 安装其他依赖库
pip install -r requirements.txt --use-pep517 -i https://mirrors.aliyun.com/pypi/simple/
```

### 3、安装TTS库

```bash
# WebUI的gradio需要Coqui TTS支持
# 如果是windows是，需要安装Visual Studio 生成工具vs_BuildTools
# 下载
https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/
# 安装
# 只选“使用C++的桌面开发”一个选项
# linux，有gcc就可以了
# 安装TTS
pip install TTS -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 4、安装PyTorch

```bash
# 校验PyTorch是否安装正常
python -c "import torch; print(torch.cuda.is_available())"
# 如果为False
# 重新安装
# 查看CUDA版本
nvidia-smi
# 根据CUDA版本查找torch安装命令
# 从 https://pytorch.org/ 首页找CUDA对应版本的安装命令
# 如果未找到，可以从Previous versinos of Torch链接
# https://pytorch.org/get-started/previous-versions/ 查找
pip uninstall torch -y
pip uninstall torchvision -y
pip uninstall torchaudio -y
# cuda必须是12.0以上
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# 校验PyTorch是否安装正常
python -c "import torch; print(torch.cuda.is_available())"
# !注意，如果运行时报以下错误：
# ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
# 则要重装Pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 5、降低gradio版本

```shell
pip install gradio==3.50.2 -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

### 6、下载模型

模型在checkpoints和gfpgan下，如已复制，不用重复下载。

#### 6.1 Linux

```bash
# （1）从github下载
bash scripts/download_models.sh
# （2）从aliendao下载
sudo apt install unzip
# 下载sadtalker
wget http://61.133.217.142:20800/download/models/sadtalker/sadtalker.zip
mkdir -p checkpoints/
unzip sadtalker.zip -d checkpoints/
# 下载gfpgan
wget http://61.133.217.142:20800/download/models/sadtalker/gfpgan.zip
unzip gfpgan.zip
```

#### 6.2 Windows

```bash
# 下载wget for winodws，复制到system32下
https://eternallybored.org/misc/wget/
# 运行git bash
在SadTalker目录下空白处，右键选Open git Bash here
# 下载模型
scripts/download_models.sh
# 或直接用https://github.com/OpenTalker/SadTalker主页上网盘下载后解压
# 下载完成后，目录结构如下：
SadTalker
├─checkpoints
│  │  mapping_00109-model.pth.tar
│  │  mapping_00229-model.pth.tar
│  │  SadTalker_V0.0.2_256.safetensors
│  │  SadTalker_V0.0.2_512.safetensors
│
├─gfpgan
│  └─weights
│          alignment_WFLW_4HG.pth
│          detection_Resnet50_Final.pth
│          GFPGANv1.4.pth
│          parsing_parsenet.pth
```

### 7、TTS/api.py修改

```shell
Linux出现TypeError: 'ModelManager' object is not subscriptable错误，windows正常，不用修改
# 需要修改TTS/api.py
# 修改方法见https://github.com/coqui-ai/TTS/issues/3429
# ~/anaconda3/envs/meta-human/lib/python3.10/site-packages/TTS或指定conda目录的TTS下
# 或用sudo find / -name TTS搜索
将api.py第126行，改为
return ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False).list_models()
```

### 8、app_sadtalker.py修改

```shell
# 1、如果在linux下，修改app_sadtalker.py第10行，避免下载TTS模型
in_webui = True
# 2、修改app_sadtalker.py最后一行为：
demo.launch(server_name='0.0.0.0')
```

### 9、启动程序

#### 9.1 WebUI

```bash
python app_sadtalker.py
```

#### 9.2 客户端方式

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --enhancer gfpgan
# 结果保存在 /时间戳/*.mp4
```

## 四、coqui-ai-TTS安装

### 1、TTS 安装

```bash
pip install TTS -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、模型下载

```shell
# 下载模型文件
# （1）从github下载
https://github.com/coqui-ai/TTS/releases/download/v0.0.10/tts_models--zh-CN--baker--tacotron2-DDC-GST.zip
# 解压到C:\Users\当前用户名\AppData\Local下
%USERPROFILE%\AppData\Local\tts\tts_models--multilingual--multi-dataset--your_tts\模型文件
# windows
%USERPROFILE%\AppData\Local\tts\tts_models--zh-CN--baker--tacotron2-DDC-GST
# linux
/home/user/.local/share/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST
# （2）从aliendao下载
wget http://61.133.217.142:20800/download/models/sadtalker/tts_models--zh-CN--baker--tacotron2-DDC-GST.zip
unzip tts_models--zh-CN--baker--tacotron2-DDC-GST.zip
mkdir -p ~/.local/share/tts
mv ./tts_models--zh-CN--baker--tacotron2-DDC-GST ~/.local/share/tts
```

### 3、运行

```bash
conda activate meta-human
python tts-test.py
```

## 五、meta-human数字人应用运行

```shell
conda activate meta-human
python meta-human.py
```

