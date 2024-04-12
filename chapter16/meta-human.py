import os
import sys
import shutil
import gradio as gr
import torch
from TTS.api import TTS
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.utils.init_path import init_path
from src.generate_facerender_batch import get_facerender_data
from src.generate_batch import get_data

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = None
preprocess_model = None
animate_from_coeff = None
audio_to_coeff = None
preprocess = 'crop'
size = 256


def load_tts_model():
    global tts
    print(TTS().list_models())
    tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST",
              progress_bar=False).to(device)


def load_sadtalker_model():
    global preprocess_model
    global animate_from_coeff
    global audio_to_coeff
    checkpoint_path = 'checkpoints'
    os.environ['TORCH_HOME'] = checkpoint_path
    config_path = 'src/config'

    sadtalker_paths = init_path(
        checkpoint_path, config_path, size, False, preprocess)
    # 根据输入的音频和姿势风格生成表情和姿势的Model
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    # 从视频或图像文件中提取关键信息的Model
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    # 生成视频并对其进行大小调整、音频处理的Model
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)


def text2audio(_text):
    output = "output.wav"
    if len(_text) < 10:
        raise gr.exceptions.Error('需要至少10个汉字')
    tts.tts_to_file(text=_text, file_path=output)
    return output


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()


def generate_video(source_image, driven_audio):
    batch_size = 1
    exp_scale = 1.0
    use_enhancer = False
    save_dir = './results/'
    # 准备目录
    os.makedirs(save_dir, exist_ok=True)
    input_dir = os.path.join(save_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    # 复制照片文件到input目录
    pic_path = os.path.join(input_dir, os.path.basename(source_image))
    shutil.copy2(source_image, input_dir)
    # 复制音频文件到input目录
    if driven_audio is not None and os.path.isfile(driven_audio):
        audio_path = os.path.join(input_dir,
                                  os.path.basename(driven_audio))
    shutil.copy2(driven_audio, input_dir)
    # 人脸识别
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = \
        preprocess_model.generate(pic_path, first_frame_dir,
                                  preprocess, True, size)
    if first_coeff_path is None:
        raise gr.exceptions.Error("未检测到人脸")
    # 从音频数据提取特征
    batch = get_data(first_coeff_path, audio_path, device,
                     ref_eyeblink_coeff_path=None, still=False,
                     idlemode=False,
                     length_of_audio=0, use_blink=True)
    coeff_path = audio_to_coeff.generate(batch, save_dir, 0, None)
    # 根据音频特征准备面部渲染的数据
    data = get_facerender_data(coeff_path, crop_pic_path,
                               first_coeff_path,
                               audio_path, batch_size, still_mode=False,
                               preprocess=preprocess,
                               size=size, expression_scale=exp_scale)
    # 根据特征生成目标视频
    return_path = animate_from_coeff.generate(
        data, save_dir,  pic_path,
        crop_info, enhancer='gfpgan' if use_enhancer else None,
        preprocess=preprocess, img_size=size)
    video_name = data['video_name']
    print(f'The generated video is named {video_name} in {save_dir}')
    torch_gc()
    return return_path


def meta_demo():
    with gr.Blocks(analytics_enabled=False) as demo_interface:
        gr.Markdown("<div align='center'> <h2>数字人演示</h2></div>")
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="source_audio"):
                    with gr.TabItem('音频输入'):
                        tts = gr.Button(
                            '文本生成音频', elem_id="audio_generate",
                            variant='primary')
                        with gr.Column(variant='panel'):

                            input_text = gr.Textbox(
                                label="文本生成音频[需要至少输入10个汉字]",
                                lines=5,
                                placeholder="请输入文本")
                    driven_audio = gr.Audio(
                        label="Audio", source="upload",
                        type="filepath")
                with gr.Tabs(elem_id="source_image"):
                    with gr.TabItem('照片输入'):
                        with gr.Column(variant='panel'):
                            source_image = gr.Image(
                                label="Image", source="upload",
                                type="filepath",
                                elem_id="image").style(width=512)
            with gr.Column(variant='panel'):
                with gr.TabItem('视频输出'):
                    with gr.Tabs(elem_id="video_area"):
                        with gr.Row():
                            submit = gr.Button(
                                '视频生成', elem_id="video_generate",
                                variant='primary')
                    with gr.Tabs(elem_id="dist_video"):
                        gen_video = gr.Video(
                            label="视频",
                            format="mp4").style(width=256)
        submit.click(
            fn=generate_video,
            inputs=[source_image,
                    driven_audio],
            outputs=[gen_video]
        )
        tts.click(fn=text2audio, inputs=[
            input_text], outputs=[driven_audio])

    return demo_interface


if __name__ == "__main__":
    load_tts_model()
    load_sadtalker_model()
    demo = meta_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0")
