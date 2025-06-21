import asyncio
import logging
from threading import Lock
from transformers import AutoTokenizer
import torch
import numpy as np
import os

from tritonclient.utils import InferenceServerException
from fastapi import FastAPI, Request
import tritonclient.http as httpclient
from pydantic import BaseModel

from backend.core.processors.preprocess import prepare_data
from backend.core.processors.postprocess import decode_emotion
from backend.core.inference.predict import predict

from prometheus_fastapi_instrumentator import Instrumentator



app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
triton_client = httpclient.InferenceServerClient(url="172.16.30.137:8000")


class EmotionRequest(BaseModel):
    sentence: str

async def predict_emotion(req: EmotionRequest) -> dict:
    try:
        # Prepare the input for the model
        sentence = req.sentence

        input_ids_tensor, attention_mask_tensor = prepare_data(sentence)

        outputs = predict(input_ids_tensor, attention_mask_tensor)

        response = {}

        if outputs["default"] is not None:
            response["default_bert"] = decode_emotion(outputs["default"][0])
        if outputs["onnx"] is not None:
            response["onnx_bert"] = decode_emotion(outputs["onnx"][0])
        if outputs["tensorrt"] is not None:
            response["tensorRT_bert"] = decode_emotion(outputs["tensorrt"][0])

        if not response:
          raise HTTPException(status_code=500, detail="All model inferences failed.")

        return response


    except InferenceServerException as e:
        logging.error(f"Inference server error: {e}")
        return None