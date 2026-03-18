# 🚇 Prédiction de l'Affluence RATP — Paris 75

Modèle de Machine Learning prédiisant le trafic journalier dans les transports parisiens (métro, bus, tram) à partir de données météo, calendaires et événementielles.

---

## 📋 Description du projet

Ce projet prédit le nombre de validations journalières dans les transports en commun du département 75 (Paris) sur le second semestre 2019 (juillet → décembre).

Le modèle apprend à partir de **12 variables explicatives** regroupées en 3 familles :

| Famille | Variables |
|---|---|
| Météo | Température, Pluie (mm), Heures de pluie, Humidité, Vent, Visibilité |
| Calendrier | Jour de la semaine, Weekend, Mois, Jour férié, Vacances scolaires (Zone C) |
| Événements | Grève (déc. 2019) |

---

## 📊 Données utilisées

| Source | Description |
|---|---|
| `Travel_titles_with_departments.csv` | Validations journalières par département — Open Data IDFM |
| `H_{dep}_2010-2019.csv.gz` | Données météo horaires par station — Météo-France (8 départements IDF) |

---

## 🛠️ Méthodologie

```
Données brutes
    │
    ├── Transport  →  agrégation journalière par département
    │
    └── Météo      →  lecture par chunks (RAM)
                   →  filtre sur la période utile
                   →  moyenne par station, puis régionale IDF
                       │
                       ▼
                Fusion & construction des features
                       │
                       ▼
         Split stratifié sur la grève (80/20)
         (garantit que les jours de grève sont
          représentés dans train ET test)
                       │
                       ▼
         RandomForestRegressor (100 arbres)
         + TimeSeriesSplit (5 folds, validation CV)
                       │
                       ▼
              Évaluation — MAE / MAPE / R²
              Permutation importance
              Sauvegarde → modele_ratp.joblib
```

---

## 🚀 Installation

```bash
pip install notebook
pip install pandas numpy scikit-learn matplotlib joblib
```

### Lancer Jupyter

```bash
python -m notebook
```

---

## 📁 Structure du projet

```
.
├── ratp_prediction.ipynb          # Notebook principal
├── modele_ratp.joblib             # Modèle entraîné (généré à l'exécution)
├── Travel_titles_with_departments.csv
├── H_75_2010-2019.csv.gz
├── H_77_2010-2019.csv.gz
├── H_78_2010-2019.csv.gz
├── H_91_2010-2019.csv.gz
├── H_92_2010-2019.csv.gz
├── H_93_2010-2019.csv.gz
├── H_94_2010-2019.csv.gz
├── H_95_2010-2019.csv.gz
└── README.md
```

---

## 📓 Structure du notebook

| Cellule | Contenu |
|---|---|
| 0 | Configuration globale & imports |
| 0b | Calendrier — jours fériés & vacances Zone C |
| 1 | Chargement des données transports |
| 2 | Chargement météo IDF (lecture par chunks) |
| 3 | Fusion & construction des features |
| 4 | Statistiques descriptives & visualisations |
| 5 | Entraînement (split stratifié + TimeSeriesSplit) |
| 6 | Évaluation — permutation importance, réel vs prédit |
| 7 | Sauvegarde & rechargement du modèle |
| 8 | Simulateur d'affluence |

---

## 🔮 Simulateur

Le notebook inclut un simulateur permettant de prédire le trafic pour n'importe quel scénario :

```python
# Charger le modèle sauvegardé
import joblib
payload  = joblib.load('modele_ratp.joblib')
modele   = payload['modele']
colonnes = payload['colonnes']

# Simuler un lundi de novembre normal
calculer_affluence(modele, colonnes,
    temp=10, pluie=0, greve=0,
    jour_type='Lundi', mois=11)
# → ~3 287 987 voyageurs

# Simuler un lundi de grève
calculer_affluence(modele, colonnes,
    temp=10, pluie=0, greve=1,
    jour_type='Lundi', mois=11)
# → ~865 121 voyageurs  (−73.7%)
```

---

## 📈 Résultats

| Métrique | Valeur |
|---|---|
| MAE (erreur absolue) | ~158 000 voyageurs/jour |
| MAPE (erreur relative) | ~9.9% |
| Score R² | 0.93 / 1.00 |

**Facteurs les plus influents** (permutation importance) :
1. Jour de la semaine
2. Jour férié
3. Weekend
4. Vacances scolaires
5. Grève

---

## ✅ Conclusion

Le modèle Random Forest atteint un **R² de 0.93**, ce qui signifie qu'il explique 93% de la variance du trafic journalier. Avec une erreur moyenne de ~158 000 voyageurs sur des journées pouvant dépasser 3 millions de validations, les prédictions sont fiables pour la grande majorité des scénarios.

Les résultats confirment que le trafic parisien est avant tout **structuré par le calendrier** : le jour de la semaine, les jours fériés et les weekends dominent largement. La météo joue un rôle secondaire mais mesurable. La grève de décembre 2019, bien que rare, provoque la chute la plus sévère observée dans les données : **−73.7% de trafic**.

Un apprentissage clé du projet : la méthode d'évaluation par défaut du Random Forest (`feature_importances_`) sous-estimait fortement l'impact de la grève en raison de sa rareté dans le dataset. Le passage à la **permutation importance** a permis de corriger ce biais et d'obtenir une image fidèle de l'influence réelle de chaque variable.

---

## 🔭 Pistes d'amélioration

| Axe | Description |
|---|---|
| **Multi-années** | Étendre les données à 2016-2018 pour capturer d'autres grèves et renforcer la saisonnalité |
| **Nouvelles features** | Ajouter les événements ponctuels (concerts, matchs, manifestations) via l'Open Data Paris |
| **Granularité horaire** | Passer de la prédiction journalière à la prédiction par heure et par ligne |
| **Modèles alternatifs** | Comparer avec XGBoost, LightGBM ou un modèle de série temporelle (Prophet, SARIMA) |
| **Déploiement** | Exposer le simulateur via une API Flask/FastAPI et une interface web interactive |

---

## ⚠️ Points de vigilance

- Le dataset couvre uniquement **6 mois** (juil.→déc. 2019) — un entraînement multi-années améliorerait la robustesse.
- La **grève de décembre 2019** est un événement rare (27 jours) : un split aléatoire standard la placerait entièrement dans le test set. Le split stratifié utilisé ici garantit sa présence dans les deux sets.
- La méthode `feature_importances_` du Random Forest **sous-estime les variables rares** comme la grève. La **permutation importance** est utilisée à la place pour une évaluation correcte.
