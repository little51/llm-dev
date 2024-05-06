import time
import torch
import uvicorn
import shortuuid
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str = "stable-code-3b"


class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]
    temperature: Optional[float] = 0.1
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class CompletionResponseChoice(BaseModel):
    index: int
    text: str


class CompletionResponse(BaseModel):
    id: Optional[str] = Field(
        default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: Optional[str] = "text_completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = 'stable-code-3b'
    choices: List[CompletionResponseChoice]


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="stable-code-3b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # 解析报文
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query
    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and \
                prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content,
                               prev_messages[i+1].content])
    # 委派predict
    generate = predict(query, history, request.model)
    # 返回EventSourceResponse
    return EventSourceResponse(generate, media_type="text/event-stream")


def predict_chunk_head(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=""),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    return chunk


def predict_chunk_content(model_id, new_content):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_content),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    return chunk


def predict_chunk_stop(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=""),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    return chunk


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_words = ['}', '###']
        stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
        return input_ids[0][-1] in stop_ids


def generate(prompt, max_new_tokens):
    global model, tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        pad_token_id=tokenizer.eos_token_id
    )
    new_response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return new_response


async def predict(query: str, history: List[List[str]], model_id: str):
    chunk = predict_chunk_head(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    current_length = 0
    token_count = 0
    max_new_tokens = 16
    while True:
        new_response = generate(query, max_new_tokens)
        new_text = new_response[current_length:]
        current_length = len(new_response)
        # push content
        chunk = predict_chunk_content(model_id, new_text)
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # stop generate
        if len(new_text) < max_new_tokens:
            break
        if str.count(new_response, '```') == 2:
            break
        if new_text in query:
            break
        token_count = token_count + max_new_tokens
        if token_count >= 512:
            break
        query = new_response
    # push stop chunk
    chunk = predict_chunk_stop(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    model_path = "./dataroot/models/stabilityai/stable-code-3b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True).cuda()
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
