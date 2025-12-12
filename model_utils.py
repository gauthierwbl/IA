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
