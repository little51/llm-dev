from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re

model = None
tokenizer = None


def load_model_tokenizer(checkpoint_path, cpu_only):
    global model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True,
    )
    if cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True,
    )


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class ChatCompletionResponse(BaseModel):
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(
        default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


def predict_chunk_head(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=""),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data],
                                   object="chat.completion.chunk")
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
                                   choice_data],
                                   object="chat.completion.chunk")
    return chunk


def predict(query: str, history: List[List[str]], model_id: str):
    global model
    global tokenizer
    # push head chunk
    chunk = predict_chunk_head(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # push content chunk
    if "voice:" in query:
        # push wait message
        chunk = predict_chunk_content(model_id, "请稍候 ")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # prepare query
        fileName = "uploads/" + query.replace("voice:", "")
        query = tokenizer.from_list_format([
            {'audio': fileName},
            {'text': '语音在说什么？'},
        ])
        # voice to text
        response, history = model.chat(tokenizer, query=query,
                                       history=None)
        chunk = predict_chunk_content(model_id, response + "\n")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # get answer with text query
        pattern = re.compile(r'\"(.*?)\"')
        result = re.search(pattern, response)
        if result:
            _text = result.group(1)
            query = tokenizer.from_list_format([
                {'text': _text},
            ])
            response, history = model.chat(
                tokenizer, query=query, history=None)
            chunk = predict_chunk_content(model_id, response + "\n")
            yield "{}".format(chunk.model_dump_json(
                exclude_unset=True))
    else:
        query = tokenizer.from_list_format([
            {'text': query},
        ])
        response, history = model.chat(tokenizer, query=query,
                                       history=None)
        chunk = predict_chunk_content(model_id, response + "\n")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # push stop chunk
    chunk = predict_chunk_stop(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    generate = predict(query, history, request.model)
    return EventSourceResponse(generate, media_type="text/event-stream")
