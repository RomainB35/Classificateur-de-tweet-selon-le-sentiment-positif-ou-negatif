curl -X POST http://127.0.0.1:8000/predict_batch   -H "Content-Type: application/json"   -d '{
        "texts": [
          "I love Cloud Run!",
          "This is terrible.",
          "BERT is awesome!"
        ]
      }'
