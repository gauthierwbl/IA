import streamlit as st
from model_utils import predict_popularity, explain_prediction

# ==========================
# CONFIGURATION DE LA PAGE
# ==========================
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# ==========================
# TITRE
# ==========================
st.title("🎧 Spotify Popularity Predictor")
st.markdown(
    "Ajuste les paramètres audio d’un morceau et observe "
    "la popularité estimée par le modèle d’intelligence artificielle."
)

st.divider()

# ==========================
# PARAMÈTRES AUDIO
# ==========================
st.subheader("🎚️ Paramètres audio")

danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
energy = st.slider("Energy", 0.0, 1.0, 0.65, 0.01)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -6.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3, 0.01)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.01, 0.01)
liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
valence = st.slider("Valence", 0.0, 1.0, 0.45, 0.01)
tempo = st.slider("Tempo (BPM)", 60, 200, 110)
duration_ms = st.slider("Duration (ms)", 60_000, 400_000, 225_000, step=1_000)

# ==========================
# GENRE MUSICAL
# ==========================
st.divider()
st.subheader("🎼 Genre musical")

GENRES = [
    "pop", "rock", "hip-hop", "edm", "dance", "house",
    "indie", "indie-pop", "electronic", "r-n-b",
    "latin", "reggaeton", "j-pop", "k-pop",
    "metal", "classical", "jazz", "blues", "country"
]

genre_choice = st.selectbox(
    "Choisis un genre",
    options=["Aucun"] + GENRES
)

genre_selected = None if genre_choice == "Aucun" else genre_choice

# ==========================
# PRÉDICTION
# ==========================
st.divider()

if st.button("🎯 Prédire la popularité"):
    input_features = {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms,
    }

    prediction = predict_popularity(input_features, genre=genre_selected)

    st.success(f"🎵 Popularité estimée : **{prediction:.1f} / 100**")

    commentary = explain_prediction(input_features, prediction)
    st.markdown(commentary)


    if prediction >= 70:
        st.markdown("🔥 **Très fort potentiel commercial**")
    elif prediction >= 50:
        st.markdown("👍 **Bon potentiel**")
    else:
        st.markdown("⚠️ **Potentiel limité**")

if st.button("⬅️ Retour à l’accueil"):
    st.switch_page("app.py")
