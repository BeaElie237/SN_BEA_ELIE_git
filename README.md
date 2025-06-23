# 📡 AI Mini Project – Classification Sonar (Roche vs Mine)

Bienvenue dans ce projet d'Intelligence Artificielle 🎯 !  
Nous allons ici construire un **pipeline complet de Machine Learning**, depuis l'entraînement d’un modèle sur le jeu de données **Sonar All Data** jusqu’à son **déploiement automatique sur Hugging Face Hub**, avec notification par email à chaque mise à jour. 🚀📩

---

## 📊 À propos du Dataset : *Sonar All Data*

- **Nom complet** : `sonar.all-data.csv`
- **Source** : UCI Machine Learning Repository
- **Taille** : 208 observations, 60 attributs + 1 label
- **But** : Prédire si un signal sonar est réfléchi par une **mine** (label `M`) ou une **roche** (label `R`)
- **Type de tâche** : Classification binaire
- **Caractéristiques** :
  - Données purement numériques (valeurs de signal)
  - Sans valeurs manquantes
  - Équilibré en classes (environ 50/50)

---

## 🧠 Objectifs du Projet

- 🧪 Charger, analyser et prétraiter le dataset
- 🏗️ Construire un **modèle IA factice** pour ce mini-projet
- 💾 Sauvegarder le modèle localement (`joblib`)
- 🔁 Déployer automatiquement sur [Hugging Face Hub](https://huggingface.co/)
- 📬 Envoyer une notification email en cas de succès du déploiement

---

## 🗃️ Structure du Dépôt

| Fichier / Dossier | Description |
|-------------------|-------------|
| `model.py` | Script contenant `train_and_save_model()` et `predict()` |
| `sonar.all-data.csv` | Jeu de données utilisé pour l'entraînement |
| `.gitignore` | Ignore les fichiers inutiles (caches, modèles, environnements virtuels) |
| `.github/workflows/deploy_hf.yml` | Workflow GitHub Actions pour le déploiement automatique |
| `README.md` | Ce fichier de documentation 📄 |

---

## ⚙️ Pipeline Technique

### 1. **Préparation**
- Chargement des données CSV
- Encodage des labels `M` et `R` en `1` et `0`
- Normalisation des variables
- Séparation entraînement/test

### 2. **Modélisation**
- Modèle utilisé : `RandomForestClassifier` (ou factice simplifié)
- Sauvegarde avec `joblib`

### 3. **Déploiement CI/CD**
- Déclenchement du pipeline à chaque `push` sur `main`
- Environnement Python 3.10 configuré
- Upload vers Hugging Face Model Hub
- Notification email via action SMTP GitHub

---

## 🛡️ Sécurité & Secrets

Utilisation des **secrets GitHub** pour protéger les informations sensibles :
- `HF_TOKEN` : pour accéder à Hugging Face
- `EMAIL_USERNAME` et `EMAIL_PASSWORD` : pour l’envoi d’emails SMTP

---

## 📤 Déploiement sur Hugging Face

🖼️ Exemple de modèle publié :  
👉 [`huggingface.co/username/sonar-ai-model`](https://huggingface.co/)

Contenu du dépôt :
- `README.md` (description du modèle)
- `model.py`
- `sonar_model.joblib`

---

## 📬 Notification Email

Une fois le modèle déployé avec succès, un **email est envoyé automatiquement** avec :
- ✅ Le lien vers le modèle Hugging Face
- 🧬 L’identifiant du commit Git
- 📅 La date du déploiement

---

## 🎥 Démo Vidéo

Une vidéo de démonstration (`demo_nom_matricule.mp4`) présente :
- Le dépôt GitHub
- Le lancement du workflow
- Le déploiement Hugging Face
- L’email reçu en temps réel

---

## 📌 Infos Étudiant

- 👨‍🎓 Réalisé par : **BEA ELIE**
- 🆔 Matricule : `XXXXXXX`
- 💼 Projet individuel - Mini-projet IA
- 🔗 GitHub : [github.com/ton-utilisateur/ai-sonar-project](https://github.com)

---

## ✅ Résultat Attendu

| Étape                  | Statut ✅ |
|------------------------|----------|
| 📁 Initialisation Git   | ✔️ |
| 🧪 Entraînement du modèle | ✔️ |
| 🚀 Déploiement Hugging Face | ✔️ |
| 📬 Notification email    | ✔️ |

---

Merci d'avoir consulté ce projet ! 🌟  
💬 Pour toute question, ouvrez une issue ou contactez-moi.  
🤝 N'hésitez pas à ⭐ ce dépôt s'il vous inspire !

