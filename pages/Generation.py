import streamlit as st

st.title("🎼 Génération musicale (IA générative)")

st.info("""
Cette fonctionnalité est en cours de développement.

Objectif :
- Décrire une musique en langage naturel
- Générer des paramètres audio optimisés
- Estimer leur potentiel de popularité
""")

if st.button("⬅️ Retour à l’accueil"):
    st.switch_page("app.py")
