# 第8章 多模态模型应用

## 一、代码获取

```shell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
git checkout bef51ae
```

## 二、安装依赖库

```bash
conda create -n stable-diffusion python=3.10 -y
conda activate stable-diffusion
# 设置全局pypi镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# windows
C:\Users\用户\AppData\Roaming\pip\pip.ini
# linux
~/.config/pip/pip.conf
# 安装依赖库
pip install -r requirements.txt
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

## 三、下载模型

```shell
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载stable-diffusion-2-1模型文件
python model_download.py --e --repo_id stabilityai/stable-diffusion-2-1 \
--token YPY8KHDQ2NAHQ2SG
# 下载的文件在./dataroot/models/stabilityai/stable-diffusion-2-1目录下
# 将模型文件移动到 stable-diffusion-webui/models/Stable-diffusion
mv ./dataroot/models/stabilityai/stable-diffusion-2-1/* models/Stable-diffusion/
```

## 四、运行服务

```shell
python launch.py --no-half --listen
```

