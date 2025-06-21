curl -X POST https://bert-fastapi-service-70236624058.europe-west1.run.app/predict   -H "Content-Type: application/json"   -d '{"text": "I love Cloud Run!"}' |jq
