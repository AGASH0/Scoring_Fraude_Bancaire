import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# 1. CHARGEMENT & PRÉPARATION DES DONNÉES
def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# 2. DÉFINITION DES PIPELINES / MODÈLES
def get_models(y_train):
    scaler = RobustScaler()
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    models = {
        "Logistic_Regression_Balanced": ImbPipeline([
            ('scaler', scaler),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        "Random_Forest_Balanced": ImbPipeline([
            ('scaler', scaler),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
        ]),
        "LightGBM_Base": ImbPipeline([
            ('scaler', scaler),
            ('classifier', LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1))
        ]),
        "LightGBM_SMOTE": ImbPipeline([
            ('scaler', scaler),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1))
        ]),
        "XGBoost_Weighted": ImbPipeline([
            ('scaler', scaler),
            ('classifier', XGBClassifier(
                scale_pos_weight=ratio,
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric="aucpr",
                n_jobs=-1
            ))
        ]),
        "CatBoost_Balanced": ImbPipeline([
            ('scaler', scaler),
            ('classifier', CatBoostClassifier(
                auto_class_weights='Balanced',
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0
            ))
        ]),
        "Balanced_Random_Forest": ImbPipeline([
            ('scaler', scaler),
            ('classifier', BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=42,
                sampling_strategy='all',
                replacement=True,
                n_jobs=-1
            ))
        ])
    }
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

    models["Ensemble_Voting_RF_LGBM"] = ImbPipeline([
        ('scaler', scaler),
        ('smote', SMOTE(random_state=42)),
        ('classifier', VotingClassifier(
            estimators=[('rf', rf), ('lgbm', lgbm)],
            voting='soft',
            weights=[1.5, 1.0]
        ))
    ])
    
    return models

# 3. ÉVALUATION PR-AUC & BENCHMARK
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    
    # Évaluation des modèles supervisés
    for name, pipeline in models.items():
        print(f"Entraînement de : {name}...")
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(y_test, y_proba)
        
        results[name] = {
            'y_proba': y_proba,
            'pr_auc': pr_auc,
            'ap': ap
        }
        print(f"-> {name} | PR-AUC: {pr_auc:.4f} | Average Precision: {ap:.4f}\n")
    
    # Évaluation du modèle non supervisé : Isolation Forest
    print("Entraînement de : Isolation_Forest (Non supervisé)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    contamination_rate = sum(y_train) / len(y_train)
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    # Score inversé : plus le score est élevé, plus c'est une anomalie
    y_scores_iso = -iso_forest.decision_function(X_test_scaled)
    # Normalisation min-max entre 0 et 1 pour la comparaison
    y_proba_iso = (y_scores_iso - y_scores_iso.min()) / (y_scores_iso.max() - y_scores_iso.min())
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba_iso)
    pr_auc_iso = auc(recall, precision)
    ap_iso = average_precision_score(y_test, y_proba_iso)
    
    results["Isolation_Forest"] = {
        'y_proba': y_proba_iso,
        'pr_auc': pr_auc_iso,
        'ap': ap_iso
    }
    print(f"-> Isolation_Forest | PR-AUC: {pr_auc_iso:.4f} | Average Precision: {ap_iso:.4f}\n")
    
    return results

# 4. MATRICE DE COÛT MÉTIER & OPTIMISATION
def compute_business_cost(y_true, y_proba, threshold, cost_fp=10.0, cost_fn=200.0):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost, fp, fn, tp

def optimize_threshold(y_true, y_proba, cost_fp=10.0, cost_fn=200.0):
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    
    for t in thresholds:
        cost, _, _, _ = compute_business_cost(y_true, y_proba, t, cost_fp, cost_fn)
        costs.append(cost)
        
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]
    default_cost, _, _, _ = compute_business_cost(y_true, y_proba, 0.50, cost_fp, cost_fn)
    
    return best_threshold, best_cost, default_cost

def print_confusion_breakdown(model_name, y_true, y_proba, threshold, cost_fp=10.0, cost_fn=200.0):
    cost, fp, fn, tp = compute_business_cost(y_true, y_proba, threshold, cost_fp, cost_fn)
    total_fraudes = sum(y_true)
    rappel = (tp / total_fraudes) * 100
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    
    print(f"--- Modèle : {model_name} (Seuil : {threshold:.3f}) ---")
    print(f"  • Fraudes détectées (Vrais Positifs) : {tp} / {total_fraudes} ({rappel:.1f} %)")
    print(f"  • Fraudes manquées (Faux Négatifs)   : {fn} (Coût : {fn * cost_fn:,.0f} €)")
    print(f"  • Fausses alertes (Faux Positifs)     : {fp} (Coût : {fp * cost_fp:,.0f} €)")
    print(f"  • Précision des alertes               : {precision:.1f} %")
    print(f"  • COÛT TOTAL RISQUE                   : {cost:,.2f} €\n")

# EXÉCUTION

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"  
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    models = get_models(y_train)
    benchmark_results = evaluate_models(models, X_train, y_train, X_test, y_test)
    
    print("=" * 65)
    print("ANALYSE COMPARATIVE DÉTAILLÉE : MATRICES DE CONFUSION")
    print("=" * 65)
    
    targets = ["Random_Forest_Balanced", "Ensemble_Voting_RF_LGBM"]
    
    for name in targets:
        proba = benchmark_results[name]['y_proba']
        best_t, min_cost, def_cost = optimize_threshold(y_test, proba)
        
        # 1. Seuil par défaut standard (0.50)
        print_confusion_breakdown(name, y_test, proba, threshold=0.50)
        # 2. Seuil optimisé financièrement
        print_confusion_breakdown(name, y_test, proba, threshold=best_t)
        print("-" * 65)