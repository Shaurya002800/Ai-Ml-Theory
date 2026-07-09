# Sentiment Classifier API

Fine-tuned DistilBERT for binary sentiment classification.
**Live demo → [HuggingFace Space](YOUR_SPACE_URL)**
**Model → [HuggingFace Hub](YOUR_MODEL_URL)**

## Results

| Method            | Trainable params | Val accuracy | Val F1 |
|-------------------|-----------------|--------------|--------|
| Full fine-tune    | 66M (100%)      | ~91%         | ~0.91  |
| LoRA r=8 (this)   | 300K (0.44%)    | ~90%         | ~0.90  |

## API

**POST /predict**
```json
{"text": "This movie was fantastic."}
→ {"sentiment": "positive", "confidence": 0.9823, "latency_ms": 87.4}
```

**POST /predict/batch**
```json
{"texts": ["Great!", "Terrible.", "Okay."]}
→ {"predictions": [...], "total_latency_ms": 124.1}
```

## Architecture
- Base model: `distilbert-base-uncased` (66M params)
- Fine-tuning: LoRA (r=8, alpha=16, target: q_lin + v_lin)
- Training data: SST-2 (Stanford Sentiment Treebank)
- Framework: HuggingFace Transformers + PEFT + FastAPI

## Key findings
- LoRA matched full fine-tune accuracy at 0.44% of trainable params
- Model struggles with sarcasm and double negatives (documented in notebook)
- Short sentences (< 5 words) have ~3% lower accuracy than longer ones

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

## What I'd do next
- Add confidence threshold — return "uncertain" below 0.65
- Fine-tune on domain-specific data (e.g. product reviews)
- Add request caching for repeated inputs