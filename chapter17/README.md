# 第17章 Prompt-gen训练及应用

## 一、词汇表生成

### 1、环境

```shell
# 建立测试目录
mkdir prompt-gen
cd prompt-gen
# 建立虚拟环境
conda create -n prompt-vocab python=3.10 -y
conda activate prompt-vocab
pip install -r requirements-vocab.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

### 2、生成

```shell
conda activate prompt-vocab
# 将data/*.txt生成train.json
python build_trainfile.py
# 按train.json生成词汇表
python make-vocab.py
```

## 二、训练

### 1、环境

```shell
conda deactivate
conda create -n prompt-train python=3.10 -y
conda activate prompt-train
pip install -r requirements-train.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
# 验证pytorch
python -c "import torch; print(torch.cuda.is_available())"
# 重装pytorch
pip uninstall torch -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# 重新验证pytorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 2、生成tokenized

```shell
conda deactivate
conda activate prompt-train
python make-tokenized.py
```

### 3、训练过程

```shell
conda deactivate
conda activate prompt-train
python train.py
```

### 4、监控训练情况

```shell
tensorboard --logdir tensorboard_summary
tensorboard --logdir tensorboard_summary --bind_all
http://127.0.0.1:6006
```

## 三、生成

### 1、命令行测试

```shell
python generate.py
```

### 2、Web测试

```shell
python web.py
```

```shell
# windows
curl -X POST http://127.0.0.1:5005/generate -H "Content-Type: application/json" -d "{\"context\":\"高血压要注意\",\"maxlength\":50,\"samples\":5}"

# linux
curl -X POST http://192.168.10.9:5005/generate -H "Content-Type: application/json" -d '{"context":"你好","maxlength":50,"samples":5}'
```

