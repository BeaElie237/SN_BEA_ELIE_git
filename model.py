"""
Script de prédiction pour les modèles SVM et KNN entraînés sur le dataset Sonar.
Charge un nouvel échantillon, le transforme comme les données d'entraînement,
et affiche les prédictions des deux modèles.
"""

import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Predictor:
    def __init__(self, model_dir='model'):
        """
        Initialise la classe en chargeant les modèles entraînés.
        
        :param model_dir: Répertoire contenant les modèles sauvegardés.
        """
        self.models = {
            'svm': joblib.load(f'{model_dir}/svm_model.pkl'),
            'knn': joblib.load(f'{model_dir}/knn_model.pkl')
        }
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_scaler(self, x_train_path='data/x_train.npy'):
        """Charge x_train et ajuste le scaler pour normaliser les nouvelles données."""
        x_train = np.load(x_train_path)
        self.scaler.fit(x_train)
        self.is_fitted = True
        print("Scaler entraîné sur les données d'entraînement.")

    def preprocess_input(self, raw_input):
        """
        Applique le prétraitement aux données d'entrée (mise à l'échelle).
        
        :param raw_input: Liste ou tableau numpy (1D) des 60 caractéristiques
        :return: Donnée prête pour la prédiction (2D)
        """
        if not self.is_fitted:
            raise RuntimeError("Le scaler n'a pas été entraîné. Appelez `fit_scaler()` d'abord.")
        
        raw_array = np.array(raw_input).reshape(1, -1)
        return self.scaler.transform(raw_array)

    def predict(self, raw_input):
        """
        Effectue une prédiction avec les deux modèles sur une donnée d'entrée.

        :param raw_input: Liste des 60 caractéristiques numériques
        :return: Dictionnaire des prédictions
        """
        input_scaled = self.preprocess_input(raw_input)
        predictions = {}

        for name, model in self.models.items():
            pred = model.predict(input_scaled)[0]
            predictions[name] = 'Mine' if pred == 1 else 'Roche'
        
        return predictions


if __name__ == "__main__":
    # Exemple réel d'entrée (ligne 0 du dataset sonar)
    sample = [
        0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
        0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
        0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
        0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
        0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
        0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032
    ]

    predictor = Predictor()
    predictor.fit_scaler()

    try:
        result = predictor.predict(sample)
        print("🔍 Prédictions pour l'échantillon :")
        for model_name, pred in result.items():
            print(f"🧠 {model_name.upper()} → {pred}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
