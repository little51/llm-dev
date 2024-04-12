import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(TTS().list_models())
tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST",
          progress_bar=False).to(device)
tts.tts_to_file(text="你好，请问你叫什么名字？", file_path="output.wav")
