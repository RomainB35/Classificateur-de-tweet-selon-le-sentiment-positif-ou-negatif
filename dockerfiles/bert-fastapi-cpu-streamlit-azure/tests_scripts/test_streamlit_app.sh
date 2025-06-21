curl -s http://localhost:8501 | grep -q '<title>' && echo "✅ Streamlit fonctionne" || echo "❌ Streamlit ne répond pas"
