curl -X POST https://bert-fastapi-service-70236624058.europe-west1.run.app/predict_batch   -H "Content-Type: application/json"   -d '{
        "texts": [
          "I love Cloud Run!",
          "This is terrible.",
          "BERT is awesome!"
        ]
      }' |jq
