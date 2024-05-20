# 大语言模型量化

## 一、llama.cpp量化

### 1、w64devkit下载

[https://github.com/skeeto/w64devkit/releases/download/v1.21.0/w64devkit-1.21.0.zip](https://github.com/skeeto/w64devkit/releases/download/v1.21.0/w64devkit-1.21.0.zip，)

### 2、代码获取

```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 19885d2
```

### 3、编译

```shell
cd llama.cpp
make -j 4
```

#### 4、建立python环境

```shell
conda create -n llama.cpp python=3.10 -y
conda activate llama.cpp
# 基本依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
# tiktoken库
pip install tiktoken -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

### 5、下载模型

```shell
# 从aliendao.cn下载Qwen/Qwen-1.8B 模型文件
wget https://aliendao.cn/model_download.py
python model_download.py --e --repo_id Qwen/Qwen-1_8B --token YPY8KHDQ2NAHQ2SG
# 下载后的文件在dataroot/models/Qwen/Qwen-1_8B 目录下
```

### 6、量化过程

```shell
# 降精度
python convert-hf-to-gguf.py dataroot/models/Qwen/Qwen-1_8B
# 量化
quantize.exe dataroot/models/Qwen/Qwen-1_8B/ggml-model-f16.gguf dataroot/models/Qwen/Qwen-1_8B/ggml-model-q5_k_m.gguf q5_k_m
```

### 7、测试

```shell
main.exe -m dataroot/models/Qwen/Qwen-1_8B/ggml-model-q5_k_m.gguf -n 512 --chatml
server.exe -m  dataroot/models/Qwen/Qwen-1_8B/ggml-model-q5_k_m.gguf -c 2048
```

## 二、 gemma.cpp量化

### 1、获取源码

```shell
git clone https://github.com/google/gemma.cpp
cd gemma.cpp
git checkout 0221956
```

### 2、编译

```shell
# 下载Visual Studio Build Tools 2022
https://aka.ms/vs/17/release/vs_BuildTools.exe
# 安装Visual Studio Build Tools
选择使用C++的桌面开发和适用于Windows的C++ Clang工具
# 设定环境变量path
# 将cmake.exe所在的目录（如c:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
）加到path路径中
# 配置build目录
cmake --preset windows
# 使用Visual Studio Build Tools编译项目
# 其中-j后面的数字用CPU的核数，用于多核编译，提高编译效率
cmake --build --preset windows -j 4
# 编译完成
编译的结果在build/Release/gemma.exe
```

### 3、量化模型下载

https://www.kaggle.com/models/google/gemma/frameworks/gemmaCpp

### 4、推理应用

```shell
build\Release\gemma.exe --tokenizer models\tokenizer.spm --compressed_weights models\2b-it-sfp.sbs --model 2b-it
```

