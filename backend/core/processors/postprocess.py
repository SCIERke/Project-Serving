import numpy as np
import torch
import torch.nn.functional as F

EMOTION_LABELS = [
    "anger", "confusion", "desire", "disgust", "fear", "guilt", "happiness",
    "love", "neutral", "sadness", "sarcasm", "shame", "surprise"
]

def decode_emotion(logits):
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    top_idx = int(np.argmax(probs))
    return {
        "label": EMOTION_LABELS[top_idx],
        "score": round(float(probs[top_idx]), 4),
        "all_scores": {EMOTION_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)}
    }
