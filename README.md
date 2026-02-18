# YBoost-IA-Data

## Installation NoteBook Jupyter

```
pip install notebook
```

## Pour lancer le NoteBook Jupyter

```
python -m notebook
```

## Pour installer scikit learn

```
py -m pip install scikit-learn
```

## Pour installer setuptools

```
python -m pip install setuptools
```

# 🚉 Prédiction de l'Affluence des Transports Parisiens (2019)

Ce projet vise à prédire l'affluence dans les stations de métro et RER à Paris en utilisant des données historiques de validation et des variables météorologiques détaillées.

## 📋 Présentation du Projet
L'objectif est d'entraîner un modèle de **Machine Learning** capable de comprendre comment les conditions climatiques (pluie, température, brouillard) influencent le nombre de voyageurs. Le projet se concentre sur le deuxième semestre de 2019 (Juillet à Décembre).

## 📊 Données utilisées
- **Données de transport** : Validations par station (Open Data IDFM).
- **Données météo** : Données horaires de Météo-France (Station Paris-Montsouris).
  - *Température* (convertie en Celsius)
  - *Précipitations* (cumul journalier)
  - *Humidité* (%)
  - *Vitesse du vent* (m/s)
  - *Visibilité*
  - *Nébulosité* (couverture nuageuse)

## 🛠️ Méthodologie (basée sur le guide ML)
Le projet suit les étapes rigoureuses du prétraitement de données :
1. **Nettoyage** : Filtrage de la période (01/07/2019 - 31/12/2019).
2. **Agrégation** : Conversion des données horaires météo en moyennes journalières pour correspondre aux validations.

## 🚀 Installation
```bash
pip install pandas numpy scikit-learn