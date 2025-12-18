import streamlit as st
import pandas as pd

from model_utils import (
    generate_audio_profile,
    evaluate_generated_profile,
    explain_prediction
)

# ==========================
# CONFIG PAGE
# ==========================
st.set_page_config(page_title="Génération musicale", page_icon="🎼")

st.title("🎼 Génération musicale (IA générative)")

st.markdown("""
Cette fonctionnalité illustre la **partie générative** du projet.

Contrairement à la prédiction, ici l’IA **ne reçoit pas une musique existante** :
elle **propose automatiquement des paramètres audio** susceptibles de produire
un morceau populaire selon les données Spotify.
""")

st.markdown("---")

# ==========================
# CHOIX UTILISATEUR
# ==========================
st.subheader("🎧 Choix du style musical")

genre = st.selectbox(
    "Sélectionnez un genre musical",
    [
        "pop", "rock", "hip-hop", "electronic", "indie",
        "jazz", "classical", "latin", "metal"
    ]
)

st.markdown("""
👉 L’IA va générer une configuration **cohérente avec ce genre**
en se basant sur les caractéristiques moyennes observées
dans les morceaux populaires du dataset.
""")

# ==========================
# PROFIL DE BASE (DATA-DRIVEN)
# ==========================
base_profile = {
    "danceability": 0.6,
    "energy": 0.65,
    "loudness": -7.0,
    "speechiness": 0.05,
    "acousticness": 0.3,
    "instrumentalness": 0.01,
    "liveness": 0.15,
    "valence": 0.45,
    "tempo": 110,
    "duration_ms": 220_000,
}

# ==========================
# GÉNÉRATION
# ==========================
if st.button("🎶 Générer une musique"):

    generated_profile = generate_audio_profile(base_profile)

    score = evaluate_generated_profile(generated_profile, genre)

    st.success(f"⭐ Popularité estimée : **{score:.1f} / 100**")

    st.markdown("### 🎛️ Paramètres audio générés")
    st.dataframe(pd.DataFrame([generated_profile]).T, use_container_width=True)

    st.markdown("### 🧠 Analyse de l’IA")
    explanation = explain_prediction(generated_profile, score)
    st.markdown(explanation)

    st.info("""
ℹ️ **Important**  
Cette génération repose uniquement sur des **statistiques du dataset**
et sur le **modèle prédictif entraîné**.

➡️ La génération par **texte libre (prompt)** sera ajoutée ensuite
à l’aide d’un **LLM (Mistral / OpenAI)**.
""")

st.markdown("---")

if st.button("⬅️ Retour à l’accueil"):
    st.switch_page("app.py")
