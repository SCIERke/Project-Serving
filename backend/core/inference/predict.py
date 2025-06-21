import tritonclient.http as httpclient
import logging


triton_client = httpclient.InferenceServerClient(url="172.16.30.137:8000")

def predict(input_ids_tensor, attention_mask_tensor) -> dict:
    outputs = {}
    try:
        outputs["default"] = triton_client.infer(
            model_name="default_bert",
            inputs=[input_ids_tensor, attention_mask_tensor]
        ).as_numpy("logits")
    except Exception as e:
        logging.error(f"default_bert failed: {e}")
        outputs["default"] = None

    try:
        outputs["onnx"] = triton_client.infer(
            model_name="onnx_bert",
            inputs=[input_ids_tensor, attention_mask_tensor]
        ).as_numpy("logits")
    except Exception as e:
        logging.error(f"onnx_bert failed: {e}")
        outputs["onnx"] = None

    try:
        outputs["tensorrt"] = triton_client.infer(
            model_name="tensorRT_bert",
            inputs=[input_ids_tensor, attention_mask_tensor]
        ).as_numpy("logits")
    except Exception as e:
        logging.error(f"tensorRT_bert failed: {e}")
        outputs["tensorrt"] = None

    return outputs