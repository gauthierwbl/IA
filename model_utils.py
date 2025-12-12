import joblib
import pandas as pd

# ==========================
# CHARGEMENT DU MODÈLE
# ==========================
MODEL_PATH = "models/best_spotify_model_xgboost.joblib"
model = joblib.load(MODEL_PATH)

# Features attendues par le modèle
EXPECTED_FEATURES = model.feature_names_in_

# Liste des genres (one-hot)
GENRE_FEATURES = [
    f for f in EXPECTED_FEATURES if f.startswith("genre_clean_")
]


# ==========================
# FONCTION DE PRÉDICTION
# ==========================
def predict_popularity(user_features: dict, genre: str | None = None) -> float:
    """
    Prédit la popularité Spotify d'un morceau.

    user_features : dict contenant les paramètres audio
    genre : str (ex: 'pop', 'rock', ...) ou None
    """

    # 1️⃣ Initialisation de toutes les features à 0
    full_input = {feature: 0 for feature in EXPECTED_FEATURES}

    # 2️⃣ Injection des features audio
    for key, value in user_features.items():
        if key in full_input:
            full_input[key] = value

    # 3️⃣ Valeur par défaut
    if "explicit" in full_input:
        full_input["explicit"] = 0

    # 4️⃣ Gestion du genre (one-hot encoding)
    if genre:
        genre_feature = f"genre_clean_{genre}"
        if genre_feature in full_input:
            full_input[genre_feature] = 1

    # 5️⃣ DataFrame final
    df = pd.DataFrame([full_input])

    # 6️⃣ Prédiction
    prediction = model.predict(df)[0]

    return float(prediction)

def explain_prediction(features: dict, score: float) -> str:
    explanations_pos = []
    explanations_neg = []

    # Heuristiques simples basées sur l'analyse dataset
    if features["danceability"] > 0.6:
        explanations_pos.append("la musique est dansante")
    elif features["danceability"] < 0.4:
        explanations_neg.append("la musique est peu dansante")

    if features["energy"] > 0.65:
        explanations_pos.append("elle est énergique")
    elif features["energy"] < 0.4:
        explanations_neg.append("elle manque d’énergie")

    if features["loudness"] > -7:
        explanations_pos.append("le volume est bien adapté aux standards actuels")
    elif features["loudness"] < -12:
        explanations_neg.append("le volume est trop faible pour ressortir")

    if features["acousticness"] > 0.5:
        explanations_pos.append("le côté acoustique peut séduire un public de niche")

    if features["instrumentalness"] > 0.5:
        explanations_neg.append("le caractère instrumental limite l’audience")

    if features["duration_ms"] < 150_000:
        explanations_neg.append("la durée est trop courte")
    elif features["duration_ms"] > 300_000:
        explanations_neg.append("la durée est un peu longue pour le streaming")

    # Construction du texte
    explanation = f"🎧 **Analyse du score ({score:.1f}/100)**\n\n"

    if explanations_pos:
        explanation += "✅ **Points positifs** :\n"
        for e in explanations_pos:
            explanation += f"- {e}\n"

    if explanations_neg:
        explanation += "\n⚠️ **Points limitants** :\n"
        for e in explanations_neg:
            explanation += f"- {e}\n"

    explanation += (
        "\n📊 **Interprétation globale** :\n"
        "Le modèle combine ces éléments pour estimer le potentiel de popularité. "
        "Un bon équilibre entre énergie, danse et durée favorise un score élevé."
    )

    return explanation
