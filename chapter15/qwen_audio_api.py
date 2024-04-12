from qwen_audio_service import load_model_tokenizer
from pathlib import Path
import os
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
from qwen_audio_service import create_chat_completion, \
    ChatCompletionRequest, ChatCompletionResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/upload")
async def upload_file(clientid: str = Form(...),
                      file: UploadFile = UploadFile(...)):
    upload_path = "./uploads"
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    try:
        fileName = f'{upload_path}/{file.filename}'
        with open(fileName, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(content={"message": "文件上传成功",
                                     "filename": file.filename})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500,
                            content={"error":
                                     f"发生错误: {str(e)}"})


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    return create_chat_completion(request)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true',
                        default=False, required=False)
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        required=True)
    args = parser.parse_args()
    load_model_tokenizer(args.checkpoint_path, args.cpu_only)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
