import torch
from transformers import AutoModelForSpeechSeq2Seq, \
    AutoProcessor, pipeline
from pydub import AudioSegment

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 \
    if torch.cuda.is_available() else torch.float32
model_id = "./dataroot/models/openai/whisper-large-v3"


def trans_m4a_to_mp3(m4a_file, mp3_file):
    song = AudioSegment.from_file(m4a_file)
    song.export(mp3_file, format='mp3')


if __name__ == "__main__":
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    trans_m4a_to_mp3("test.m4a", "test.mp3")
    result = pipe("test.mp3")
    print(result["text"])
