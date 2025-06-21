import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn

app = FastAPI()

# Chargement du modèle et tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")
model.eval()

class InputText(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {"message": "Service BERT FastAPI is running."}

@app.post("/predict")
def predict(input_text: InputText):
    inputs = tokenizer(input_text.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence = probs.max().item()
        prediction = torch.argmax(probs, dim=1).item()
        pred_remapped = 0 if prediction == 0 else 4
        sentiment = "Tweet négatif" if pred_remapped == 0 else "Tweet positif"
        return {
            "tweet": input_text.text,
            "prediction": pred_remapped,
            "confidence": confidence,
            "sentiment": sentiment
        }

@app.post("/predict_batch")
def predict_batch(batch: BatchInput):
    inputs = tokenizer(batch.texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidences = probs.max(dim=1).values.tolist()
        predictions = torch.argmax(probs, dim=1).tolist()
        predictions_remapped = [0 if p == 0 else 4 for p in predictions]
        sentiments = ["Tweet négatif" if p == 0 else "Tweet positif" for p in predictions_remapped]

    results = [
        {
            "tweet": text,
            "prediction": pred,
            "confidence": conf,
            "sentiment": sent
        }
        for text, pred, conf, sent in zip(batch.texts, predictions_remapped, confidences, sentiments)
    ]
    return results

