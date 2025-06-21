import streamlit as st
import requests
import json
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# --- Configuration Azure Application Insights ---
CONNECTION_STRING = (
    "InstrumentationKey=YOUR-INSTRUMENTATION-KEY"
    "IngestionEndpoint=YOUR-INGESTION-ENDPOINT"
    "LiveEndpoint=YOUR-LIVE-ENDPOINT"
    "ApplicationId=YOUR-APPLICATION-ID"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Éviter les doublons de handler (important avec Streamlit)
if not any(isinstance(handler, AzureLogHandler) for handler in logger.handlers):
    logger.addHandler(AzureLogHandler(connection_string=CONNECTION_STRING))

st.title("🔍 Test du modèle BERT exposé avec FastAPI")
st.write("Saisissez un tweet pour obtenir une prédiction de sentiment.")

# Initialisation de la session_state pour stocker les données
if 'tweet_input' not in st.session_state:
    st.session_state['tweet_input'] = ''

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'confidence' not in st.session_state:
    st.session_state['confidence'] = None

if 'feedback' not in st.session_state:
    st.session_state['feedback'] = None

# Saisie du tweet
tweet_input = st.text_area("✍️ Votre tweet ici :", height=100, value=st.session_state['tweet_input'])

# Bouton pour envoyer à l'API
if st.button("📤 Envoyer à l'API"):
    if not tweet_input.strip():
        st.warning("Veuillez entrer un tweet avant d'envoyer.")
    else:
        payload = {
            "text": tweet_input
        }
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            prediction_data = response.json()
            st.session_state['tweet_input'] = tweet_input
            st.session_state['prediction'] = prediction_data["sentiment"]
            st.session_state['confidence'] = round(prediction_data["confidence"] * 100, 2)
            st.session_state['feedback'] = None  # reset feedback

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel API : {e}")

# Affichage de la prédiction si elle existe
if st.session_state['prediction'] is not None:
    st.success(f"✅ Prédiction : `{st.session_state['prediction']}` avec {st.session_state['confidence']}% de confiance")

    # Feedback utilisateur
    feedback = st.radio("Est-ce correct ?", ["✅ Oui", "❌ Non"], index=0 if st.session_state['feedback'] == "✅ Oui" else 1 if st.session_state['feedback'] == "❌ Non" else None, key="feedback_radio")

    if st.button("Valider le feedback"):
        st.session_state['feedback'] = feedback

        log_data = {
            "tweet": st.session_state['tweet_input'],
            "prediction": st.session_state['prediction'],
            "confidence": f"{st.session_state['confidence']}"
        }

        if feedback == "✅ Oui":
            logger.info("✅ Prédiction validée", extra={"custom_dimensions": log_data})
            st.success("🎉 Merci pour la validation !")
        elif feedback == "❌ Non":
            logger.warning("❌ Tweet mal prédit", extra={"custom_dimensions": log_data})
            st.info("⚠️ Signalement envoyé à Application Insights")

