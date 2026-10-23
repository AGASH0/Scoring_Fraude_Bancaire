"""
Détection de fraude bancaire : benchmark de 15 modèles et optimisation du seuil par coût métier.

Jeu de données : Kaggle "Credit Card Fraud Detection" (creditcard.csv).

Protocole (sans fuite d'information) :
    1. Suppression des doublons, puis split train / test (aléatoire stratifié, ou temporel).
       Le test n'est utilisé qu'une seule fois, à la fin.
    2. Validation croisée stratifiée sur le train : sélection du modèle par Average Precision (AP).
    3. Seuil de décision choisi sur les prédictions out-of-fold (OOF) du train, jamais sur le test.
    4. Modèle final réentraîné sur tout le train, évalué sur le test avec le seuil figé,
       avec intervalle de confiance bootstrap sur le coût.

Usage :
    python fraud_detection_benchmark.py --data data/creditcard.csv --folds 5 --out results
    python fraud_detection_benchmark.py --data data/creditcard.csv --folds 3 --models Random_Forest_Balanced,LightGBM_Weighted
    python fraud_detection_benchmark.py --data data/creditcard.csv --split temporal

Remarque : avec --split temporal, le test correspond aux 20 % de transactions les plus récentes,
mais la validation croisée sur le train reste aléatoire (limite à mentionner dans le README).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
THRESHOLDS = np.linspace(0.001, 0.999, 999)


# ==============================================================================
# 1. DONNÉES
# ==============================================================================
def load_data(path: str):
    """Charge le CSV, supprime les doublons et construit les variables."""
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{n_before - len(df)} doublons supprimés ({len(df)} transactions conservées)")

    time = df["Time"].copy()
    # 'Time' = secondes depuis le début de la période : non généralisable.
    # On le remplace par l'heure de la journée, qui a un sens métier.
    df["Hour"] = (df["Time"] / 3600) % 24
    df = df.drop(columns="Time")

    y = df.pop("Class")
    return df, y, df["Amount"].copy(), time


# ==============================================================================
# 2. MODÈLES (une "factory" par modèle : tout est recréé à chaque fold)
# ==============================================================================
def _ratio(y) -> float:
    """Rapport négatifs / positifs sur le jeu d'entraînement."""
    return float((y == 0).sum() / (y == 1).sum())


class IsolationForestScorer:
    """Isolation Forest non supervisé, score d'anomalie ramené dans [0, 1] (interface predict_proba)."""

    def __init__(self, n_estimators: int = 150):
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        self.scaler_ = RobustScaler()
        Xs = self.scaler_.fit_transform(X)
        self.iso_ = IsolationForest(n_estimators=self.n_estimators, random_state=SEED, n_jobs=-1).fit(Xs)
        s = -self.iso_.decision_function(Xs)
        self.lo_, self.hi_ = float(s.min()), float(s.max())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = -self.iso_.decision_function(self.scaler_.transform(X))
        s = np.clip((s - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)
        return np.column_stack([1 - s, s])


class AmountWeightedLGBM:
    """
    LightGBM entraîné avec une pondération par le coût : chaque fraude pèse proportionnellement
    à son montant (cost-sensitive learning), pour aligner l'entraînement sur la fonction de coût métier.
    """

    def fit(self, X, y):
        y_arr = np.asarray(y)
        fraud = y_arr == 1
        amt = X["Amount"].to_numpy(dtype=float)
        base = np.sqrt((~fraud).sum() / fraud.sum())
        w = np.ones(len(y_arr))
        w[fraud] = base * np.maximum(amt[fraud], 1.0) / max(amt[fraud].mean(), 1.0)
        self.model_ = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
        ).fit(X, y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_logreg(y):
    # La mise à l'échelle n'est utile que pour les modèles linéaires.
    return ImbPipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])


def make_rf(y):
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    )


def make_balanced_rf(y):
    return BalancedRandomForestClassifier(
        n_estimators=200, sampling_strategy="all", replacement=True,
        random_state=SEED, n_jobs=-1,
    )


def make_extra_trees(y):
    return ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    )


def make_lgbm_base(y):
    # Référence "sans traitement du déséquilibre".
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED,
                          n_jobs=-1, verbose=-1)


def make_lgbm_weighted(y):
    # Pondération modérée (racine du ratio) : un poids plein (~500) rend les probabilités extrêmes.
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, scale_pos_weight=float(np.sqrt(_ratio(y))),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def make_lgbm_smote(y):
    # SMOTE dans le pipeline imblearn : appliqué uniquement sur les données d'entraînement.
    return ImbPipeline([
        ("smote", SMOTE(random_state=SEED)),
        ("clf", LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED,
                               n_jobs=-1, verbose=-1)),
    ])


def make_xgb(y):
    return XGBClassifier(
        scale_pos_weight=_ratio(y), n_estimators=200, learning_rate=0.05, max_depth=5,
        tree_method="hist", eval_metric="aucpr", random_state=SEED, n_jobs=-1,
    )


def make_catboost(y):
    return CatBoostClassifier(
        auto_class_weights="Balanced", iterations=200, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )


def make_voting(y):
    # Pas de SMOTE ici : les deux modèles gèrent déjà le déséquilibre par pondération
    # (SMOTE + class_weight = double correction). Poids égaux : pas de réglage arbitraire.
    return VotingClassifier(
        estimators=[("rf", make_rf(y)), ("lgbm", make_lgbm_weighted(y))],
        voting="soft",
    )


def make_stacking(y):
    # Méta-modèle (régression logistique) entraîné sur des prédictions out-of-fold (cv=3).
    return StackingClassifier(
        estimators=[("rf", make_rf(y)), ("lgbm", make_lgbm_weighted(y)), ("lr", make_logreg(y))],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=3,
    )


def make_rf_calibrated(y):
    # Calibration sigmoïde (plus stable que l'isotonique quand les fraudes sont rares).
    return CalibratedClassifierCV(make_rf(y), method="sigmoid", cv=3)


FACTORIES = {
    "Logistic_Regression_Balanced": make_logreg,
    "Random_Forest_Balanced": make_rf,
    "Balanced_Random_Forest": make_balanced_rf,
    "Extra_Trees_Balanced": make_extra_trees,
    "LightGBM_Base": make_lgbm_base,
    "LightGBM_Weighted": make_lgbm_weighted,
    "LightGBM_SMOTE": make_lgbm_smote,
    "LightGBM_Amount_Weighted": lambda y: AmountWeightedLGBM(),
    "XGBoost_Weighted": make_xgb,
    "CatBoost_Balanced": make_catboost,
    "Voting_RF_LGBM": make_voting,
    "Stacking_RF_LGBM_LR": make_stacking,
    "Random_Forest_Calibrated": make_rf_calibrated,
    "Isolation_Forest": lambda y: IsolationForestScorer(),
}


# ==============================================================================
# 3. VALIDATION CROISÉE (AP par fold + probabilités out-of-fold)
# ==============================================================================
def cross_validate_model(factory, X, y, n_splits: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_ap = []
    for tr, va in skf.split(X, y):
        model = factory(y.iloc[tr])
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        fold_ap.append(average_precision_score(y.iloc[va], proba))
    return oof, float(np.mean(fold_ap)), float(np.std(fold_ap))


# ==============================================================================
# 4. COÛT MÉTIER ET MÉTRIQUES D'EXPLOITATION
# ==============================================================================
def cost_curve(y_true, proba, thresholds, fn_costs, cost_fp: float) -> np.ndarray:
    """
    Coût total pour chaque seuil (calcul exact et vectorisé).
        - Faux positif : alerte injustifiée = cost_fp (frais de traitement, friction client)
        - Faux négatif : fraude manquée = montant de la transaction (fn_costs)
        - Vrais positifs / négatifs : 0
    Une transaction est signalée si proba >= seuil.
    Repère théorique : pour des probabilités calibrées, seuil optimal = c_FP / (c_FP + c_FN).
    """
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    p_legit = np.sort(proba[~y_true])
    p_fraud = proba[y_true]
    fn_c = np.asarray(fn_costs, dtype=float)[y_true]

    order = np.argsort(p_fraud)
    p_fraud_sorted = p_fraud[order]
    cum_fn = np.concatenate([[0.0], np.cumsum(fn_c[order])])

    n_fp = len(p_legit) - np.searchsorted(p_legit, thresholds, side="left")
    fn_cost = cum_fn[np.searchsorted(p_fraud_sorted, thresholds, side="left")]
    return n_fp * cost_fp + fn_cost


def cost_at(y_true, proba, threshold: float, fn_costs, cost_fp: float) -> float:
    return float(cost_curve(y_true, proba, np.array([threshold]), fn_costs, cost_fp)[0])


def best_threshold(y_true, proba, fn_costs, cost_fp: float) -> float:
    costs = cost_curve(y_true, proba, THRESHOLDS, fn_costs, cost_fp)
    return float(THRESHOLDS[np.argmin(costs)])


def precision_recall_at_k(y_true, proba, k: int):
    """Précision et rappel si les analystes ne peuvent traiter que les k transactions les plus suspectes."""
    y_arr = np.asarray(y_true)
    top = np.argsort(-np.asarray(proba), kind="stable")[:k]
    tp = int(y_arr[top].sum())
    return tp / k, tp / int(y_arr.sum())


def bootstrap_savings(y_true, proba, fn_costs, threshold, cost_fp, n_boot, seed=SEED):
    """IC à 95 % (bootstrap stratifié) du coût et de l'économie vs 'sans modèle' sur le test."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    p_arr, c_arr = np.asarray(proba), np.asarray(fn_costs, dtype=float)
    pos, neg = np.where(y_arr == 1)[0], np.where(y_arr == 0)[0]
    costs, savings = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, pb, cb = y_arr[idx], p_arr[idx], c_arr[idx]
        base = cb[yb == 1].sum()
        c = cost_at(yb, pb, threshold, cb, cost_fp)
        costs.append(c)
        savings.append(1 - c / base)
    return np.percentile(costs, [2.5, 97.5]), np.percentile(savings, [2.5, 97.5])


# ==============================================================================
# 5. GRAPHIQUES
# ==============================================================================
def plot_pr_curves(oof_store, y_train, bench, path: Path, top: int = 6):
    plt.figure(figsize=(7, 5))
    for name in bench["model"].head(top):
        p, r, _ = precision_recall_curve(y_train, oof_store[name])
        plt.plot(r, p, label=f"{name} (AP={average_precision_score(y_train, oof_store[name]):.3f})")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"Courbes précision-rappel, top {top} (prédictions out-of-fold)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cost_curves(cost_oof, cost_test, t_opt, path: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(THRESHOLDS, cost_oof, label="Train (out-of-fold) : sert à choisir le seuil")
    plt.plot(THRESHOLDS, cost_test, label="Test : évaluation finale")
    plt.axvline(t_opt, color="red", linestyle="--", label=f"Seuil retenu = {t_opt:.3f}")
    plt.axvline(0.5, color="grey", linestyle=":", label="Seuil standard 0.5")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Coût total (€)")
    plt.title("Coût métier en fonction du seuil")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ==============================================================================
# 6. PROGRAMME PRINCIPAL
# ==============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="all",
                    help="'all' ou liste séparée par des virgules. Disponibles : " + ", ".join(FACTORIES))
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--cost-fp", type=float, default=10.0, help="Coût d'une fausse alerte (€)")
    ap.add_argument("--fixed-fn-cost", type=float, default=None,
                    help="Coût fixe d'une fraude manquée (€). Par défaut : montant réel de la transaction.")
    ap.add_argument("--alerts-k", type=int, default=100, help="Capacité de traitement pour la précision@k")
    ap.add_argument("--bootstrap", type=int, default=300, help="Nombre de rééchantillonnages pour l'IC")
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    if args.models == "all":
        names = list(FACTORIES)
    else:
        names = [n.strip() for n in args.models.split(",")]
        unknown = [n for n in names if n not in FACTORIES]
        if unknown:
            raise SystemExit(f"Modèles inconnus : {unknown}\nDisponibles : {list(FACTORIES)}")

    # --- Données et split -------------------------------------------------------
    X, y, amounts, time = load_data(args.data)
    fn_costs = amounts if args.fixed_fn_cost is None else pd.Series(args.fixed_fn_cost, index=amounts.index)

    if args.split == "temporal":
        order = np.argsort(time.to_numpy(), kind="stable")
        cut = int(len(order) * 0.8)
        tr_idx, te_idx = order[:cut], order[cut:]
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED)

    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)
    c_tr, c_te = fn_costs.iloc[tr_idx].to_numpy(), fn_costs.iloc[te_idx].to_numpy()
    print(f"Split {args.split} | Train : {len(y_tr)} ({y_tr.sum()} fraudes) "
          f"| Test : {len(y_te)} ({y_te.sum()} fraudes)\n")

    # --- Benchmark par validation croisée ---------------------------------------
    rows, oof_store = [], {}
    for name in names:
        print(f"Validation croisée : {name}...", flush=True)
        oof, ap_mean, ap_std = cross_validate_model(FACTORIES[name], X_tr, y_tr, args.folds)
        oof_store[name] = oof
        rows.append({
            "model": name, "AP_cv_mean": ap_mean, "AP_cv_std": ap_std,
            "Brier_oof": brier_score_loss(y_tr, oof),
        })
        print(f"  -> AP = {ap_mean:.4f} +/- {ap_std:.4f}", flush=True)

    bench = pd.DataFrame(rows).sort_values("AP_cv_mean", ascending=False).reset_index(drop=True)
    bench.to_csv(out / "benchmark_cv.csv", index=False)
    print("\n", bench.round(4).to_string(index=False))
    print("(Brier : indicatif, seuls les modèles calibrés sont comparables sur ce critère)")

    best_name = bench.loc[0, "model"]
    tied = bench[bench["AP_cv_mean"] >= bench.loc[0, "AP_cv_mean"] - bench.loc[0, "AP_cv_std"]]["model"].tolist()
    print(f"\nMeilleur modèle (AP en CV) : {best_name}")
    if len(tied) > 1:
        print(f"Statistiquement proches (écart < 1 écart-type) : {', '.join(tied)}")

    # --- Seuil choisi sur le train (OOF), jamais sur le test ---------------------
    oof_best = oof_store[best_name]
    t_opt = best_threshold(y_tr, oof_best, c_tr, args.cost_fp)
    t_ref = args.cost_fp / (args.cost_fp + float(c_tr[y_tr.to_numpy() == 1].mean()))

    # --- Évaluation finale unique sur le test ------------------------------------
    final_model = FACTORIES[best_name](y_tr)
    final_model.fit(X_tr, y_tr)
    p_te = final_model.predict_proba(X_te)[:, 1]
    ap_test = average_precision_score(y_te, p_te)

    cost_none = float(c_te[y_te.to_numpy() == 1].sum())  # aucune alerte : toutes les fraudes passent
    cost_std = cost_at(y_te, p_te, 0.5, c_te, args.cost_fp)
    cost_opt = cost_at(y_te, p_te, t_opt, c_te, args.cost_fp)
    cost_ci, sav_ci = bootstrap_savings(y_te, p_te, c_te, t_opt, args.cost_fp, args.bootstrap)

    tn, fp, fn, tp = confusion_matrix(y_te, (p_te >= t_opt).astype(int)).ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    k = min(args.alerts_k, len(y_te))
    p_at_k, r_at_k = precision_recall_at_k(y_te, p_te, k)

    print("\n" + "=" * 68)
    print("RÉSULTATS FINAUX SUR LE JEU DE TEST (modèle et seuil choisis sur le train)")
    print("=" * 68)
    print(f"Modèle                              : {best_name}")
    print(f"Average Precision (test)            : {ap_test:.4f}")
    print(f"Seuil retenu (OOF train)            : {t_opt:.3f}  (repère théorique si calibré : {t_ref:.3f})")
    print(f"Précision / rappel au seuil retenu  : {precision:.3f} / {recall:.3f}")
    print(f"Fraudes détectées / manquées / fausses alertes : {tp} / {fn} / {fp}")
    print(f"Précision@{k} / rappel@{k}            : {p_at_k:.3f} / {r_at_k:.3f}")
    print(f"Coût sans modèle                    : {cost_none:,.2f} €")
    print(f"Coût au seuil standard (0.50)       : {cost_std:,.2f} €")
    print(f"Coût au seuil retenu                : {cost_opt:,.2f} € (IC95 % : {cost_ci[0]:,.0f} à {cost_ci[1]:,.0f} €)")
    print(f"Économie vs sans modèle             : {(1 - cost_opt / cost_none) * 100:.1f} % "
          f"(IC95 % : {sav_ci[0] * 100:.1f} à {sav_ci[1] * 100:.1f} %)")
    print(f"Économie vs seuil 0.50              : {(1 - cost_opt / cost_std) * 100:.1f} %")

    # --- Sensibilité aux hypothèses de coût -------------------------------------
    sens = []
    for c_fp in (5, 10, 20, 50):
        t = best_threshold(y_tr, oof_best, c_tr, c_fp)
        sens.append({
            "cost_fp": c_fp, "seuil": round(t, 3),
            "cout_test": round(cost_at(y_te, p_te, t, c_te, c_fp), 2),
            "cout_sans_modele": round(cost_none, 2),
        })
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensibilite_cout_fp.csv", index=False)
    print("\nSensibilité au coût d'une fausse alerte :\n", sens.to_string(index=False))

    # --- Sauvegardes -------------------------------------------------------------
    plot_pr_curves(oof_store, y_tr, bench, out / "pr_curves.png")
    plot_cost_curves(
        cost_curve(y_tr, oof_best, THRESHOLDS, c_tr, args.cost_fp),
        cost_curve(y_te, p_te, THRESHOLDS, c_te, args.cost_fp),
        t_opt, out / "cost_vs_threshold.png",
    )
    metrics = {
        "best_model": best_name, "split": args.split, "ap_test": float(ap_test), "threshold": t_opt,
        "precision": float(precision), "recall": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        f"precision_at_{k}": float(p_at_k), f"recall_at_{k}": float(r_at_k),
        "cost_no_model": cost_none, "cost_threshold_0.5": cost_std, "cost_optimal": cost_opt,
        "cost_ci95": [float(cost_ci[0]), float(cost_ci[1])],
        "savings_vs_no_model_ci95": [float(sav_ci[0]), float(sav_ci[1])],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nFichiers enregistrés dans : {out.resolve()}")
    mask_fraud = (y_te.to_numpy() == 1)

    # Identification des Vrais Positifs (TP) et Faux Négatifs (FN)
    tp_mask = mask_fraud & (p_te >= t_opt)
    fn_mask = mask_fraud & (p_te < t_opt)

    # Extraction des montants réels
    montants_tp = c_te[tp_mask]
    montants_fn = c_te[fn_mask]

    # Calcul des statistiques financières
    total_fraude = c_te[mask_fraud].sum()
    total_tp = montants_tp.sum()
    total_fn = montants_fn.sum()

    print("\n" + "=" * 60)
    print("AUDIT DE L'IMPACT FINANCIER DES FRAUDES (JEU DE TEST)")
    print("=" * 60)
    print(f"Perte totale théorique (sans modèle) : {total_fraude:,.2f} €")
    print(f"Montant total intercepté (TP)        : {total_tp:,.2f} € ({total_tp / total_fraude * 100:.1f} % des pertes sauvées)")
    print(f"Montant total manqué (FN)            : {total_fn:,.2f} € ({total_fn / total_fraude * 100:.1f} % des pertes résiduelles)\n")

    print("--- Comparaison statistique des montants unitaires ---")
    print(f"Fraudes interceptées ({len(montants_tp)} TP) :")
    print(f"  • Médiane : {np.median(montants_tp):.2f} €")
    print(f"  • Moyenne : {np.mean(montants_tp):.2f} €")
    print(f"  • Max     : {np.max(montants_tp):.2f} €")

    print(f"\nFraudes manquées ({len(montants_fn)} FN) :")
    print(f"  • Médiane : {np.median(montants_fn):.2f} €")
    print(f"  • Moyenne : {np.mean(montants_fn):.2f} €")
    print(f"  • Max     : {np.max(montants_fn):.2f} €")

    # Boîte à moustaches comparative
    df_plot = pd.DataFrame({
        'Montant': np.concatenate([montants_tp, montants_fn]),
        'Statut': [f'Fraude Interceptée ({len(montants_tp)} TP)'] * len(montants_tp) + 
                  [f'Fraude Manquée ({len(montants_fn)} FN)'] * len(montants_fn)
    })

    plt.figure(figsize=(7, 4))
    import seaborn as sns
    sns.boxplot(data=df_plot, x='Statut', y='Montant', palette=['#2ecc71', '#e74c3c'])
    plt.title("Distribution des montants : Fraudes interceptées vs Fraudes manquées")
    plt.ylabel("Montant de la transaction (€)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(out / "montants_fraude_tp_vs_fn.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
    