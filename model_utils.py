import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel
from typing import Optional
from transformers import MusicgenForConditionalGeneration, AutoProcessor

load_dotenv()

# ==========================
# CHARGEMENT DU MODÈLE
# ==========================

MODEL_PATH = "models/best_spotify_model_xgboost.joblib"
model = joblib.load(MODEL_PATH)

# Features attendues par le modèle (ordre strict)
EXPECTED_FEATURES = model.feature_names_in_

# Genres encodés en one-hot
GENRE_FEATURES = [
    f for f in EXPECTED_FEATURES if f.startswith("genre_clean_")
]

# Features audio numériques principales
NUMERIC_AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

# ==========================
# PRÉDICTION DE POPULARITÉ
# ==========================

def predict_popularity(user_features: dict, genre: str | None = None) -> float:
    """
    Prédit la popularité Spotify d'un morceau à partir de ses paramètres audio.
    """

    # Initialisation de toutes les features à 0
    full_input = {feature: 0 for feature in EXPECTED_FEATURES}

    # Injection des paramètres audio
    for key, value in user_features.items():
        if key in full_input:
            full_input[key] = value

    # Valeur par défaut pour explicit
    if "explicit" in full_input:
        full_input["explicit"] = 0

    # Gestion du genre (one-hot encoding)
    if genre:
        genre_feature = f"genre_clean_{genre}"
        if genre_feature in full_input:
            full_input[genre_feature] = 1

    df = pd.DataFrame([full_input])

    prediction = model.predict(df)[0]
    return float(prediction)

# ==========================
# EXPLICATION HEURISTIQUE
# ==========================

def explain_prediction(features: dict, score: float) -> str:
    """
    Fournit une explication qualitative du score de popularité.
    """

    positives, negatives = [], []

    if features["danceability"] > 0.6:
        positives.append("la musique est dansante")
    elif features["danceability"] < 0.4:
        negatives.append("la musique est peu dansante")

    if features["energy"] > 0.65:
        positives.append("elle est énergique")
    elif features["energy"] < 0.4:
        negatives.append("elle manque d’énergie")

    if features["loudness"] > -7:
        positives.append("le volume est adapté aux standards actuels")
    elif features["loudness"] < -12:
        negatives.append("le volume est trop faible")

    if features["duration_ms"] < 150_000:
        negatives.append("la durée est trop courte")
    elif features["duration_ms"] > 300_000:
        negatives.append("la durée est longue pour le streaming")

    text = f"🎧 **Analyse du score ({score:.1f}/100)**\n\n"

    if positives:
        text += "✅ **Points positifs** :\n" + "\n".join(f"- {p}" for p in positives) + "\n\n"

    if negatives:
        text += "⚠️ **Points limitants** :\n" + "\n".join(f"- {n}" for n in negatives) + "\n\n"

    text += (
        "📊 **Interprétation globale** :\n"
        "Le score reflète l’équilibre entre énergie, danse, durée et lisibilité sonore."
    )

    return text

# ==========================
# GÉNÉRATION DE PARAMÈTRES
# ==========================

def generate_audio_profile(base_profile: dict) -> dict:
    """
    Génère une configuration audio candidate (copie contrôlée).
    """

    generated = base_profile.copy()

    # Ajout de légères variations réalistes
    generated["danceability"] = np.clip(generated["danceability"] + np.random.normal(0, 0.05), 0, 1)
    generated["energy"] = np.clip(generated["energy"] + np.random.normal(0, 0.05), 0, 1)
    generated["tempo"] = np.clip(generated["tempo"] + np.random.normal(0, 5), 60, 180)

    return generated


def evaluate_generated_profile(profile: dict, genre: str) -> float:
    """
    Évalue une configuration générée via le modèle prédictif.
    """
    return predict_popularity(profile, genre)



# =====================================
# GÉNÉRATION DE RECOMMANDATIONS + AUDIO
# =====================================

class MusicParams(BaseModel):
    genre: Optional[str] = None
    tempo: Optional[int] = None
    key: Optional[str] = None
    duration: Optional[float] = None

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"))

def extract_parameters(user_input):
    llm = get_llm()
    try:
        return llm.with_structured_output(MusicParams).invoke(f"Extrait en JSON : {user_input}").model_dump()
    except:
        return {k: None for k in ["genre", "tempo", "key", "duration"]}

def get_market_stats(genre=None, tempo=None):
    df = pd.read_csv('archive/dataset.csv')
    top = df[df.popularity >= df.popularity.quantile(0.8)]
    
    f = top[top.track_genre == genre] if genre and len(top[top.track_genre == genre]) > 10 else top
    if tempo:
        t_f = f[(f.tempo >= tempo-10) & (f.tempo <= tempo+10)]
        f = t_f if len(t_f) >= 5 else top
        
    res = f[['danceability', 'energy', 'key', 'mode', 'valence', 'tempo', 'duration_ms']].median().to_dict()
    if tempo: res['tempo'] = tempo
    return res, len(f)

def generate_music_audio(p_in, p_opt):
    model_id = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)

    genre = p_in['genre'] if p_in['genre'] else 'pop'
    description = f"{genre} music, mood: {p_opt['valence']*100:.0f}% happy, {p_opt['tempo']:.0f} BPM"

    inputs = processor(text=[description], padding=True, return_tensors="pt")
    audio_values = model.generate(**inputs, max_new_tokens=256)
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values.squeeze().detach().cpu().numpy().astype(np.float32)
    
    return audio_data, sampling_rate

def get_composition_advice(p_in, p_opt, key_name):
    llm = get_llm()
    prompt = f"Donne 3 conseils courts (Rythme, Accords, Ambiance) pour : {p_in['genre']}, {p_opt['tempo']:.0f} BPM, {key_name}."
    return llm.invoke(prompt).content
