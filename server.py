import os
import time
import torch
import uvicorn
import uuid
import logging
from fastapi import FastAPI, HTTPException
# Remove: from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse, JSONResponse # Ensure StreamingResponse is imported
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Literal
from contextlib import asynccontextmanager
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from threading import Thread
# Pillow, base64, io, requests remain the same
from PIL import Image
import base64
import io
import requests
import json # Need json library for dumping

# --- Configuration --- (Keep as before)
MODEL_DIR = "<your local path>/google/gemma-3-27b-it"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
DEVICE_MAP = "auto"
MODEL_DTYPE = torch.bfloat16
MAX_TOKENS = 16*1024
PORT = 8000
HOST = "0.0.0.0"

# --- Logging --- (Keep as before)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables --- (Keep as before)
model = None
processor = None

# --- Model Loading Function --- (Keep as before)
def load_model():
    # ... (same code as before) ...
    global model, processor
    logger.info(f"Starting model loading from {MODEL_DIR}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            device_map=DEVICE_MAP,
            torch_dtype=MODEL_DTYPE,
        ).eval()
        logger.info("Model loaded successfully.")
        logger.info(f"Model loaded on devices: {model.hf_device_map}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model from {MODEL_DIR}") from e

# --- FastAPI Lifespan Management --- (Keep as before)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (same code as before) ...
    logger.info("Application startup...")
    load_model()
    yield
    logger.info("Application shutdown...")
    global model, processor
    if model is not None: del model
    if processor is not None: del processor
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="Gemma 3 OpenAI-Compatible API")

# --- Pydantic Models --- (Keep as before)
# TextContentBlock, ImageUrlDetail, ImageContentBlock, ContentType
# OpenAIMessage, OpenAIChatCompletionRequest
# ChatCompletionMessage, Choice, UsageInfo, OpenAIChatCompletionResponse
# DeltaMessage, ChunkChoice, OpenAIChatCompletionChunk
# ... (all Pydantic models remain unchanged) ...
class TextContentBlock(BaseModel):
    type: Literal["text"]
    text: str

class ImageUrlDetail(BaseModel):
    url: str

class ImageContentBlock(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlDetail

ContentType = Union[TextContentBlock, ImageContentBlock]

class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentType]]

class OpenAIChatCompletionRequest(BaseModel):
    model: str = MODEL_DIR
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = MAX_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_DIR
    choices: List[Choice]
    usage: UsageInfo

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None

class ChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class OpenAIChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_DIR
    choices: List[ChunkChoice]


# --- Helper Functions --- (Keep as before)
def process_image(image_content: ImageUrlDetail) -> Image.Image:
    # ... (same code as before) ...
    url = image_content.url
    if url.startswith("data:image"):
        try:
            header, encoded = url.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError("Invalid base64 image data")
    else:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error downloading image from URL {url}: {e}")
            raise ValueError(f"Could not download image from {url}")

def map_openai_to_gemma_messages(openai_messages: List[OpenAIMessage]) -> List[Dict[str, any]]:
    # ... (same code as before) ...
    gemma_messages = []
    for msg in openai_messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            gemma_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            gemma_content_list = []
            for block in content:
                if block.type == "text":
                    gemma_content_list.append({"type": "text", "text": block.text})
                elif block.type == "image_url":
                    try:
                        image = process_image(block.image_url)
                        gemma_content_list.append({"type": "image", "image": image})
                    except ValueError as e:
                        logger.warning(f"Skipping image due to processing error: {e}")
                    except Exception as e:
                         logger.error(f"Unexpected error processing image: {e}", exc_info=True)

            if gemma_content_list:
                gemma_messages.append({"role": role, "content": gemma_content_list})
        else:
            logger.warning(f"Unsupported content type for role {role}: {type(content)}. Skipping message.")
    return gemma_messages


# --- API Endpoints ---

@app.get("/v1/models")
async def list_models():
    # ... (same code as before) ...
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": MODEL_DIR,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user",
        }]
    })

@app.post("/v1/chat/completions")
async def create_chat_completion(request: OpenAIChatCompletionRequest):
    # ... (Initial checks and message processing remain the same) ...
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        gemma_messages = map_openai_to_gemma_messages(request.messages)
        if not gemma_messages:
             raise HTTPException(status_code=400, detail="No valid messages provided or processed.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error processing input messages: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during message mapping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during message processing.")

    try:
        inputs = processor.apply_chat_template(
            gemma_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to tokenize input: {e}")

    generate_kwargs = {
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0 or request.top_p < 1.0,
        "temperature": request.temperature if request.temperature > 0 else None,
        "top_p": request.top_p if request.top_p < 1.0 else None,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    # --- Streaming Response ---
    if request.stream:
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs_stream = {**inputs, **generate_kwargs, "streamer": streamer}

        thread = Thread(target=model.generate, kwargs=generation_kwargs_stream)
        thread.start()

        # *** MODIFIED STREAM GENERATOR ***
        async def stream_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            completion_tokens = 0
            finish_reason = None

            # Yield initial chunk with role
            chunk = OpenAIChatCompletionChunk(
                id=request_id, created=created_time, model=request.model,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)]
            )
            # Manually format the SSE message
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            # Yield content chunks
            try:
                for delta_token in streamer:
                    if delta_token: # Ensure not empty string
                        completion_tokens += 1 # Approximation
                        chunk = OpenAIChatCompletionChunk(
                            id=request_id, created=created_time, model=request.model,
                            choices=[ChunkChoice(index=0, delta=DeltaMessage(content=delta_token), finish_reason=None)]
                        )
                        # Manually format the SSE message
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                # Wait for the generation thread to finish to determine the correct finish reason
                thread.join()
                finish_reason = "stop" # Assume stop for now

            except Exception as e:
                logger.error(f"Error during streaming generation: {e}", exc_info=True)
                # You might want to send an error chunk here, but let's keep it simple
                finish_reason = "error" # Or some other indicator

            # Yield final chunk with finish reason
            chunk = OpenAIChatCompletionChunk(
                id=request_id, created=created_time, model=request.model,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(), finish_reason=finish_reason)]
            )
            # Manually format the SSE message
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            # Send DONE signal
            yield "data: [DONE]\n\n"

        # *** RETURN StreamingResponse ***
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Non-Streaming Response --- (Keep as before)
    else:
        try:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generate_kwargs)

            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            decoded_text = processor.decode(generated_ids, skip_special_tokens=True)

            completion_tokens = len(generated_ids)
            total_tokens = prompt_tokens + completion_tokens

            finish_reason = "length" if completion_tokens >= request.max_tokens else "stop"

            response = OpenAIChatCompletionResponse(
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content=decoded_text.strip()),
                        finish_reason=finish_reason
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            return JSONResponse(content=response.model_dump(exclude_unset=True))

        except Exception as e:
            logger.error(f"Error during non-streaming generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")


# --- Main Execution --- (Keep as before)
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
