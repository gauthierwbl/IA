import streamlit as st
import pandas as pd
from model_utils import extract_parameters, get_market_stats, generate_music_audio, get_composition_advice

st.set_page_config(page_title="Génération Créative", page_icon="🎼")
st.title("🎼 Assistant de Création Musicale")

user_query = st.text_area("Décrivez la musique que vous souhaitez créer :", placeholder="Ex: Un rock énergique à 120 BPM...")

if st.button("🪄 Analyser et Recommander"):
    if user_query:
        with st.spinner("Analyse sémantique et statistique..."):
            p_in = extract_parameters(user_query)
            p_opt, n_titles = get_market_stats(p_in['genre'], p_in['tempo'])
            
            # Stockage en session pour la génération audio plus tard
            st.session_state.p_in = p_in
            st.session_state.p_opt = p_opt

            keys_fr = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
            key_name = f"{keys_fr[int(p_opt['key'])]} {'Majeur' if p_opt['mode'] > 0.5 else 'Mineur'}"

            st.subheader("📊 Paramètres recommandés")
            col1, col2, col3 = st.columns(3)
            col1.metric("Genre détecté", p_in['genre'].capitalize() if p_in['genre'] else "Standard")
            col2.metric("Tempo idéal", f"{p_opt['tempo']:.0f} BPM")
            col3.metric("Tonalité", key_name)

            st.markdown(f"**Analyse basée sur {n_titles} morceaux populaires.**")
            
            st.subheader("💡 Conseils de composition")
            advice = get_composition_advice(p_in, p_opt, key_name)
            st.info(advice)

if "p_opt" in st.session_state:
    st.divider()
    st.subheader("🎹 Génération de l'extrait")
    if st.button("🔊 Générer l'audio (MusicGen)"):
        with st.spinner("Synthèse audio en cours (environ 30s)..."):
            audio_data, rate = generate_music_audio(st.session_state.p_in, st.session_state.p_opt)
            st.audio(audio_data, sample_rate=rate)