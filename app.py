import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Chargement des modèles
svm_model = joblib.load('model/svm_model.pkl')
knn_model = joblib.load('model/knn_model.pkl')

# Chargement du scaler (basé sur les données d'entraînement)
x_train = np.load('data/x_train.npy')
scaler = StandardScaler()
scaler.fit(x_train)

st.set_page_config(page_title="Prédiction Sonar", layout="centered")

st.title("📡 Prédiction IA – Sonar Roche vs Mine")
st.markdown("""
Cette application utilise des modèles SVM et KNN pour prédire si un signal sonar provient d'une **Mine** ou d'une **Roche**.

- Les données doivent contenir **60 colonnes numériques**
- Vous pouvez **saisir manuellement** ou **téléverser un fichier CSV**
""")

# 📥 Méthode d'entrée
method = st.radio("Choisissez votre méthode d'entrée :", ["Saisie manuelle", "Téléverser un fichier CSV"])

# Saisie manuelle
if method == "Saisie manuelle":
    values = []
    st.subheader("Entrée des 60 valeurs sonar")
    cols = st.columns(6)
    for i in range(60):
        col = cols[i % 6]
        val = col.number_input(f"x{i+1}", value=0.02, step=0.01, format="%.4f")
        values.append(val)
    data_input = [values]

# Téléversement de fichier CSV
else:
    uploaded_file = st.file_uploader("Téléversez un fichier CSV (1 ligne = 1 échantillon)", type=["csv"])
    data_input = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != 60:
                st.error("⚠️ Le fichier doit contenir exactement 60 colonnes.")
            else:
                data_input = df.values.tolist()
                st.success("✅ Données chargées avec succès.")
        except Exception as e:
            st.error(f"❌ Erreur de lecture : {e}")

# Lancer la prédiction
if st.button("🔍 Lancer la prédiction") and data_input:
    results = []
    for row in data_input:
        x_scaled = scaler.transform(np.array(row).reshape(1, -1))
        svm_pred = svm_model.predict(x_scaled)[0]
        knn_pred = knn_model.predict(x_scaled)[0]

        results.append({
            "SVM": "Mine" if svm_pred == 1 else "Roche",
            "KNN": "Mine" if knn_pred == 1 else "Roche"
        })

    st.subheader("📊 Résultats de la prédiction")
    result_df = pd.DataFrame(results)
    st.table(result_df)
