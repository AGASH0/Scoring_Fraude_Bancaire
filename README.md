# 💳 Benchmark de Détection de Fraude Bancaire & Optimisation du Risque (Cost-Sensitive Learning)

Ce projet propose un benchmark méthodologique et étanche de **8 architectures de Machine Learning** (supervisées, d'ensemble et non supervisées) appliquées à la détection de transactions frauduleuses sur un historique hautement déséquilibré. 

Au-delà de la performance algorithmique pure, ce projet démontre comment l'alignement d'un modèle statistique sur une **matrice de coûts financiers asymétriques** permet de réduire les pertes économiques d'un établissement financier de plus de **41 %**.

---

## 📌 Problématique Métier

Dans un contexte bancaire réel, la détection de fraude fait face à deux défis structurels majeurs :
1. **L'asymétrie extrême des données** : Les fraudes représentent moins de 0,2 % des transactions. L'exactitude (*Accuracy*) ou la courbe ROC sont des métriques trompeuses car elles sont artificiellement gonflées par la classe majoritaire légitime.
2. **L'asymétrie des coûts d'erreur** :
   * **Faux Positif (FP)** : Bloquer une carte légitime génère un coût de traitement et une friction client modérée (~10 €).
   * **Faux Négatif (FN)** : Laisser passer une fraude entraîne une perte sèche directe pour la banque (~200 €).

---

## 📊 Jeu de Données

* **Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Volume** : 284 807 transactions.
* **Taux de fraude** : 492 fraudes (**0,172 %**).
* **Variables** : 28 composantes principales anonymisées ($V_1$ à $V_{28}$) issues d'une ACP, complétées par le montant (`Amount`) et le temps (`Time`).

---

## ⚙️ Méthodologie & Bonnes Pratiques

* **Cloisonnement strict (No Data Leakage)** : Découpage stratifié train/test (80/20). Les étapes de normalisation (`RobustScaler`) et de sur-échantillonnage synthétique (`SMOTE`) sont encapsulées dans des pipelines `imblearn` appliqués exclusivement sur le jeu d'entraînement.
* **Métrique de référence** : Priorité donnée à la surface sous la courbe Précision-Rappel (**PR-AUC**) et à l'**Average Precision (AP)**.
* **Cost-Sensitive Learning** : Modélisation d'une fonction de coût métier et balayage continu du seuil de décision $p \in [0.01, 0.99]$ pour identifier le point de décision économique optimal.

---

## 🏆 Résultats du Benchmark

### 1. Synthèse globale des architectures testées

| Architecture | Type | PR-AUC | Average Precision | Commentaire technique |
| :--- | :--- | :---: | :---: | :--- |
| **Ensemble_Voting_RF_LGBM** | Assemblage (Soft Voting) | **0.8732** | **0.8733** | **Meilleure performance statistique globale.** |
| **Random_Forest_Balanced** | Bagging pondéré | **0.8666** | **0.8629** | Meilleure calibration sur la queue de distribution. |
| **LightGBM_SMOTE** | Boosting + Sur-échantillonnage | 0.8344 | 0.8324 | Forte progression par rapport à la version de base. |
| **XGBoost_Weighted** | Boosting régularisé | 0.8310 | 0.8311 | Régularisation L1/L2 native via `scale_pos_weight`. |
| **CatBoost_Balanced** | Arbres symétriques | 0.8042 | 0.8049 | Arbres rigides face aux petits îlots d'anomalies. |
| **Balanced_Random_Forest** | Under-bagging natif | 0.7964 | 0.7654 | Rapide mais perte d'information sur les transactions légitimes. |
| **Logistic_Regression_Balanced** | Modèle linéaire pondéré | 0.7600 | 0.7156 | Baseline interprétable et auditable par le régulateur. |
| **LightGBM_Base** | Boosting sans correction | 0.4299 | 0.2564 | Échec critique : écrasé par la classe majoritaire. |
| **Isolation_Forest** | Détection d'anomalies non supervisée | 0.1623 | 0.1692 | Filet de secours (détection de fraudes inédites zero-day). |

---

### 2. Arbitrage Économique : Random Forest vs Voting

Pour une matrice de coûts fixée à **10 € par Faux Positif** et **200 € par Faux Négatif** sur le jeu de test (56 962 transactions dont 98 fraudes) :

```text
--- Random Forest Balanced (Seuil standard : 0.500) ---
  • Fraudes détectées (Rappel) : 78 / 98 (79.6 %)
  • Fraudes manquées (FN)      : 20 (Coût : 4 000 €)
  • Fausses alertes (FP)       : 9  (Coût : 90 €)
  • Coût total du risque       : 4 090,00 €

--- Random Forest Balanced (Seuil optimisé : 0.158) ---
  • Fraudes détectées (Rappel) : 88 / 98 (89.8 %)
  • Fraudes manquées (FN)      : 10 (Coût : 2 000 €)
  • Fausses alertes (FP)       : 39 (Coût : 390 €)
  • Coût total du risque       : 2 390,00 €  ==> -41.56 % de réduction des pertes

--- Ensemble Voting RF + LightGBM (Seuil optimisé : 0.347) ---
  • Fraudes détectées (Rappel) : 88 / 98 (89.8 %)
  • Fraudes manquées (FN)      : 10 (Coût : 2 000 €)
  • Fausses alertes (FP)       : 63 (Coût : 630 €)
  • Coût total du risque       : 2 630,00 €