"""Données et modèles statistiques de l'application.

Ce module ne contient aucun code d'interface : toute la logique statistique
(nettoyage, LDA, ACP, K-means, comparaison de modèles, seuils par maximum de
vraisemblance) est ici, et `app.py` ne s'occupe que de l'affichage.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLONNES = ["Grossesses", "Glucose", "Pression", "Peau", "Insuline", "IMC", "Pedigree", "Age", "Classe"]
VARIABLES = COLONNES[:-1]
# Dans Pima, un 0 sur ces variables signifie "valeur manquante"
ZEROS_MANQUANTS = ["Glucose", "Pression", "Peau", "Insuline", "IMC"]
GRAINE = 42


# ----------------------------------------------------------------------
# 1. Données
# ----------------------------------------------------------------------
def charger_donnees():
    """Charge Pima, remplace les 0 aberrants par la médiane de la variable."""
    df = pd.read_csv(URL, names=COLONNES)
    df[ZEROS_MANQUANTS] = df[ZEROS_MANQUANTS].replace(0, np.nan)
    df[ZEROS_MANQUANTS] = df[ZEROS_MANQUANTS].fillna(df[ZEROS_MANQUANTS].median())
    return df


# ----------------------------------------------------------------------
# 2. Modèle final (LDA + ACP) utilisé pour le diagnostic
# ----------------------------------------------------------------------
def entrainer_final(df):
    """Entraîne scaler, LDA et ACP sur toutes les données."""
    X, y = df[VARIABLES], df["Classe"]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    lda = LinearDiscriminantAnalysis().fit(X_scaled, y)
    pca = PCA(n_components=len(VARIABLES)).fit(X_scaled)
    return scaler, lda, pca, X_scaled


def niveau_risque(proba, seuil):
    """Faible / modéré / élevé selon le seuil d'alerte choisi."""
    if proba >= seuil:
        return "élevé"
    if proba >= seuil / 2:
        return "modéré"
    return "faible"


def contributions_lda(lda, patient_scaled):
    """Contribution de chaque variable au score discriminant du patient.

    Les variables étant standardisées, 0 = la moyenne de la population.
    Une valeur positive pousse vers « diabétique », une valeur négative vers « sain ».
    """
    return pd.Series(lda.coef_[0] * patient_scaled[0], index=VARIABLES).sort_values()


# ----------------------------------------------------------------------
# 3. Évaluation sur jeu de test
# ----------------------------------------------------------------------
def evaluer_lda(df):
    """LDA entraînée sur 75 % des données, évaluée sur 25 %."""
    X, y = df[VARIABLES], df["Classe"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=GRAINE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    lda = LinearDiscriminantAnalysis().fit(scaler.transform(X_train), y_train)
    proba = lda.predict_proba(scaler.transform(X_test))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "y_test": y_test.to_numpy(),
        "proba": proba,
        "auc": roc_auc_score(y_test, proba),
        "fpr": fpr,
        "tpr": tpr,
    }


def sensibilite_specificite(y_true, proba, seuil):
    """Sensibilité et spécificité pour un seuil de décision donné."""
    pred = proba >= seuil
    vrais_pos = np.sum(pred & (y_true == 1))
    vrais_neg = np.sum(~pred & (y_true == 0))
    sensibilite = vrais_pos / max(np.sum(y_true == 1), 1)
    specificite = vrais_neg / max(np.sum(y_true == 0), 1)
    return sensibilite, specificite


def courbe_seuils(y_true, proba):
    """Sensibilité et spécificité pour une grille de seuils de 0,05 à 0,95."""
    seuils = np.linspace(0.05, 0.95, 91)
    valeurs = [sensibilite_specificite(y_true, proba, s) for s in seuils]
    return pd.DataFrame(
        {"Seuil": seuils, "Sensibilité": [v[0] for v in valeurs], "Spécificité": [v[1] for v in valeurs]}
    )


# ----------------------------------------------------------------------
# 4. Comparaison de modèles (validation croisée)
# ----------------------------------------------------------------------
def comparer_modeles(df):
    """Validation croisée stratifiée à 5 plis pour 3 modèles."""
    X, y = df[VARIABLES], df["Classe"]
    modeles = {
        "LDA": LinearDiscriminantAnalysis(),
        "Régression logistique": LogisticRegression(max_iter=1000),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=200, random_state=GRAINE),
    }
    scoring = {"Accuracy": "accuracy", "AUC": "roc_auc", "Sensibilité": "recall"}
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)

    lignes = []
    for nom, modele in modeles.items():
        res = cross_validate(make_pipeline(StandardScaler(), modele), X, y, cv=plis, scoring=scoring)
        ligne = {"Modèle": nom}
        for metrique in scoring:
            valeurs = res[f"test_{metrique}"]
            ligne[metrique] = valeurs.mean()
            ligne[f"{metrique} (écart-type)"] = valeurs.std()
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("Modèle")


# ----------------------------------------------------------------------
# 5. Groupes de patients (K-means sur le premier plan factoriel)
# ----------------------------------------------------------------------
def clusteriser(X_pca_2d, k):
    return KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(X_pca_2d)


def resume_groupes(df, etiquettes):
    """Effectif, taux de diabète et profil moyen de chaque groupe."""
    return (
        df.assign(Groupe=etiquettes + 1)
        .groupby("Groupe")
        .agg(
            Effectif=("Classe", "size"),
            Pct_diabetiques=("Classe", lambda s: 100 * s.mean()),
            Glucose=("Glucose", "mean"),
            IMC=("IMC", "mean"),
            Age=("Age", "mean"),
        )
        .round(1)
    )


# ----------------------------------------------------------------------
# 6. Seuils d'alerte personnalisés (maximum de vraisemblance)
# ----------------------------------------------------------------------
def seuils_mle(df, variable):
    """Seuil d'alerte par tranche d'âge pour une variable donnée.

    Dans chaque tranche, la variable est modélisée par une loi normale par classe
    dont les paramètres sont estimés par maximum de vraisemblance
    (moyenne empirique, écart-type avec division par n). Le seuil est la valeur
    pour laquelle la probabilité a posteriori d'être diabétique atteint 50 %.
    """
    tranches = pd.cut(df["Age"], bins=[20, 30, 40, 100], labels=["21-30 ans", "31-40 ans", "41 ans et +"])
    lignes = []
    for tranche in tranches.cat.categories:
        sous = df[tranches == tranche]
        x0 = sous.loc[sous["Classe"] == 0, variable]
        x1 = sous.loc[sous["Classe"] == 1, variable]
        if len(x0) < 5 or len(x1) < 5:
            continue
        mu0, s0 = norm.fit(x0)  # estimateurs du maximum de vraisemblance
        mu1, s1 = norm.fit(x1)
        pi1 = len(x1) / len(sous)

        seuil = np.nan
        if mu1 > mu0:
            grille = np.linspace(mu0, mu1, 2000)
            d1 = pi1 * norm.pdf(grille, mu1, s1)
            d0 = (1 - pi1) * norm.pdf(grille, mu0, s0)
            posterieure = d1 / (d1 + d0)
            ok = np.where(posterieure >= 0.5)[0]
            if len(ok) > 0 and posterieure[0] < 0.5:
                seuil = grille[ok[0]]

        lignes.append(
            {
                "Tranche d'âge": tranche,
                "Effectif": len(sous),
                "Moyenne (sains)": mu0,
                "Moyenne (diabétiques)": mu1,
                "Écart-type (sains)": s0,
                "Écart-type (diabétiques)": s1,
                "Seuil d'alerte": seuil,
            }
        )
    return pd.DataFrame(lignes).round(2)
