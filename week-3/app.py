from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from transformers import pipeline
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Classifier API",
    description="DistilBERT fine-tuned on SST-2 via LoRA. "
                "Classifies text as positive or negative.",
    version="1.0.0",
)

# ── Load model at startup — once, not per request ──────────────────────────
MODEL_PATH = "./distilbert-lora-merged"

@app.on_event("startup")
async def load_model():
    global classifier
    logger.info("Loading model...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=-1,            # CPU; swap to 0 for GPU
        return_all_scores=True,
    )
    logger.info("Model loaded.")

# ── Schemas ────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=512,
                      example="This film was absolutely wonderful.")

class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32,
                             example=["Great movie!", "Terrible waste of time."])

class PredictionOut(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float

class BatchOut(BaseModel):
    predictions: List[PredictionOut]
    total_latency_ms: float

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict", response_model=PredictionOut)
def predict(body: TextInput):
    try:
        t0      = time.perf_counter()
        results = classifier(body.text, truncation=True, max_length=128)
        ms      = (time.perf_counter() - t0) * 1000

        scores    = {r["label"]: r["score"] for r in results[0]}
        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]

        label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
        return PredictionOut(
            text=body.text,
            sentiment=label_map.get(sentiment, sentiment),
            confidence=round(confidence, 4),
            latency_ms=round(ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(body: BatchInput):
    try:
        t0      = time.perf_counter()
        results = classifier(body.texts, truncation=True,
                             max_length=128, batch_size=16)
        ms      = (time.perf_counter() - t0) * 1000

        label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
        preds = []
        for text, res in zip(body.texts, results):
            scores     = {r["label"]: r["score"] for r in res}
            sentiment  = max(scores, key=scores.get)
            preds.append(PredictionOut(
                text=text,
                sentiment=label_map.get(sentiment, sentiment),
                confidence=round(scores[sentiment], 4),
                latency_ms=round(ms / len(body.texts), 2),
            ))
        return BatchOut(predictions=preds, total_latency_ms=round(ms, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))