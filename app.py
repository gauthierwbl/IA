import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Spotify Popularity AI",
    page_icon="🎧",
    layout="centered"
)

# ======================
# EN-TÊTE
# ======================
st.title("🎧 Spotify Popularity AI")
st.markdown(
    """
    **Une application d’intelligence artificielle dédiée à l’analyse et à la conception de musiques populaires.**  
    Elle combine **machine learning prédictif** et **IA générative** pour comprendre *pourquoi* une musique fonctionne et *comment* en concevoir une avec un fort potentiel de popularité.
    """
)

st.markdown("---")

# ======================
# CONTEXTE & OBJECTIFS
# ======================
st.markdown(
    """
    ### 🧠 Que fait cette application ?
    
    Cette application repose sur un **modèle de machine learning entraîné sur des données Spotify**  
    (danceability, energy, tempo, loudness, genres, etc.).
    
    Elle permet :
    - 📊 **d’estimer la popularité potentielle d’un morceau**
    - 🔍 **d’expliquer les facteurs qui influencent cette popularité**
    - 🎼 **de générer des paramètres musicaux optimisés à partir d’une description textuelle**
    
    👉 L’objectif est de **rendre l’IA compréhensible**, pas seulement performante.
    """
)

st.markdown("---")

# ======================
# BLOCS FONCTIONNALITÉS
# ======================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🔮 Prédiction de popularité")
    st.markdown(
        """
        Analysez un morceau **à partir de ses caractéristiques audio** :
        
        - danceability  
        - énergie  
        - tempo  
        - durée  
        - genre musical  
        
        L’IA :
        - prédit un **score de popularité (0–100)**  
        - explique **les points forts et les points faibles**  
        - fournit une **interprétation claire du potentiel du morceau**
        """
    )
    st.markdown("")  # espace visuel
    if st.button("👉 Accéder à la prédiction", use_container_width=True):
        st.switch_page("pages/prediction.py")

with col2:
    st.markdown("### 🎼 Génération musicale")
    st.markdown(
        """
        Décrivez une musique **en langage naturel** :
        
        > *« Une pop énergique, joyeuse, faite pour les playlists estivales »*
        
        L’IA :
        - interprète votre intention artistique  
        - traduit le texte en **paramètres audio concrets**
        - propose une **recette musicale cohérente et optimisée**
        
        *(Idéal pour la conception, l’idéation ou l’expérimentation musicale.)*
        """
    )
    st.markdown("")  # espace visuel
    if st.button("👉 Accéder à la génération", use_container_width=True):
        st.switch_page("pages/generation.py")

st.markdown("---")

# ======================
# PIED DE PAGE
# ======================
st.markdown(
    """
    🧪 **Projet IA — Analyse & Génération musicale**  
    Machine Learning · Explainability · IA générative · Streamlit
    
    *L’IA ne remplace pas la créativité — elle l’augmente.*
    """
)
