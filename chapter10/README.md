# 第10章 编程大模型应用

## 一、准备环境

```shell
# 新建测试目录
mkdir stable-code
cd stable-code
# 建立虚拟环境
conda create -n stable-code python=3.10 -y
conda activate stable-code
```

## 二、安装依赖库

```bash
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
python -c "import torch; print(torch.cuda.is_available())"
```

## 三、下载模型

```shell
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载stable-code-3b模型文件
python model_download.py --e --repo_id stabilityai/stable-code-3b --token YPY8KHDQ2NAHQ2SG
```

## 四、运行服务

```shell
python code-api.py
```

