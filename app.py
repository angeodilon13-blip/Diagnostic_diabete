"""Interface Streamlit : assistant d'aide à la décision pour le dépistage du diabète.

Organisation du projet :
    app.py            -> interface (ce fichier)
    graphiques.py     -> figures Plotly
    modeles.py        -> données et méthodes statistiques
    .streamlit/config.toml -> thème
    requirements.txt
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

import graphiques as g
import modeles as m

st.set_page_config(page_title="Aide à la décision - Diabète", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------
# Style (CSS)
# ----------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem;}
.entete {
    background: linear-gradient(135deg, #0EA5A4 0%, #2563EB 100%);
    border-radius: 18px; padding: 22px 28px; margin-bottom: 14px; color: white;
}
.entete h1 {margin: 0; padding: 0; font-size: 1.9rem; color: white;}
.entete p {margin: 6px 0 0 0; opacity: 0.92;}
.resultat {
    border-radius: 18px; padding: 22px 26px; color: white; margin-bottom: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}
.resultat h2 {margin: 0; padding: 0; font-size: 1.7rem; color: white;}
.resultat p {margin: 8px 0 0 0; opacity: 0.95;}
div[data-testid="stMetric"] {
    background: rgba(148, 163, 184, 0.10);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px; padding: 14px 18px;
}
</style>
"""

NIVEAUX = {
    "faible": {
        "icone": "✅", "degrade": "linear-gradient(135deg, #16A34A, #15803D)",
        "conduite": "Profil rassurant : maintenir une bonne hygiène de vie et un contrôle régulier.",
    },
    "modéré": {
        "icone": "🟠", "degrade": "linear-gradient(135deg, #F59E0B, #D97706)",
        "conduite": "Zone de vigilance : une glycémie à jeun et un avis professionnel sont conseillés à moyen terme.",
    },
    "élevé": {
        "icone": "⚠️", "degrade": "linear-gradient(135deg, #EF4444, #B91C1C)",
        "conduite": "Risque élevé : consulter rapidement un professionnel de santé pour confirmer par des examens.",
    },
}

LIBELLES = {
    "Grossesses": ("Nombre de grossesses", "int"),
    "Glucose": ("Glucose (mg/dL)", "int"),
    "Pression": ("Pression artérielle (mm Hg)", "int"),
    "Peau": ("Pli cutané (mm)", "int"),
    "Insuline": ("Insuline (mu U/ml)", "int"),
    "IMC": ("Indice de masse corporelle", "float"),
    "Pedigree": ("Score génétique (Pedigree)", "float"),
    "Age": ("Âge (années)", "int"),
}


def afficher(fig):
    """Affiche une figure Plotly dans une carte, sans barre d'outils."""
    with st.container(border=True):
        st.plotly_chart(fig, config={"displayModeBar": False})


# ----------------------------------------------------------------------
# Chargement et entraînement (mis en cache)
# ----------------------------------------------------------------------
@st.cache_data
def charger():
    return m.charger_donnees()


@st.cache_resource
def modele_final(df):
    return m.entrainer_final(df)


@st.cache_data
def evaluation(df):
    return m.evaluer_lda(df)


@st.cache_data
def comparaison(df):
    return m.comparer_modeles(df)


@st.cache_resource
def groupes(X_pca_2d, k):
    return m.clusteriser(X_pca_2d, k)


# ----------------------------------------------------------------------
# Barre latérale : formulaire du patient
# ----------------------------------------------------------------------
def barre_laterale(df):
    st.sidebar.markdown("### 🧑‍⚕️ Dossier du patient")
    with st.sidebar.form("formulaire_patient"):
        valeurs = {}
        for var in m.VARIABLES:
            libelle, typ = LIBELLES[var]
            mini, maxi, med = df[var].min(), df[var].max(), df[var].median()
            if typ == "int":
                valeurs[var] = st.number_input(
                    libelle, min_value=int(mini), max_value=int(maxi), value=int(med), step=1
                )
            else:
                pas = 0.01 if var == "Pedigree" else 0.1
                valeurs[var] = st.number_input(
                    libelle, min_value=float(mini), max_value=float(maxi),
                    value=float(round(med, 2)), step=pas, format="%.2f",
                )
        st.markdown("**Réglage de l'alerte**")
        seuil = st.number_input(
            "Seuil d'alerte (probabilité)", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            format="%.2f",
            help="Plus le seuil est bas, plus on détecte de malades (sensibilité), au prix de fausses alertes.",
        )
        st.form_submit_button("Analyser le patient", type="primary")
    return pd.DataFrame([valeurs], columns=m.VARIABLES), seuil


# ----------------------------------------------------------------------
# Onglets
# ----------------------------------------------------------------------
def onglet_diagnostic(patient_scaled, lda, pca, X_pca, y, patient_pca, seuil, ev):
    proba = lda.predict_proba(patient_scaled)[0][1]
    niveau = m.niveau_risque(proba, seuil)
    infos = NIVEAUX[niveau]
    sens, spec = m.sensibilite_specificite(ev["y_test"], ev["proba"], seuil)

    gauche, droite = st.columns([1, 1.2])
    with gauche:
        st.markdown(
            f"""<div class="resultat" style="background: {infos['degrade']};">
                <h2>{infos['icone']} Risque {niveau}</h2>
                <p>{infos['conduite']}</p>
            </div>""",
            unsafe_allow_html=True,
        )
        afficher(g.jauge(proba * 100, seuil))
        c1, c2 = st.columns(2)
        c1.metric("Sensibilité du modèle", f"{sens * 100:.0f} %", help="Part des malades détectés à ce seuil.")
        c2.metric("Spécificité du modèle", f"{spec * 100:.0f} %", help="Part des sains correctement reconnus.")
        st.caption(f"Niveau « modéré » entre {seuil / 2:.2f} et {seuil:.2f}. "
                   "Résultat indicatif, à confirmer par un professionnel de santé.")
    with droite:
        afficher(g.plan_factoriel(X_pca, y, patient_pca, pca.explained_variance_ratio_))
        afficher(g.contributions(m.contributions_lda(lda, patient_scaled)))


def onglet_performance(ev, seuil):
    pred = (ev["proba"] >= seuil).astype(int)
    sens, spec = m.sensibilite_specificite(ev["y_test"], ev["proba"], seuil)
    st.subheader(f"Qualité du modèle sur le jeu de test (seuil = {seuil:.2f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{(pred == ev['y_test']).mean() * 100:.1f} %")
    c2.metric("AUC", f"{ev['auc']:.3f}")
    c3.metric("Sensibilité", f"{sens * 100:.1f} %")
    c4.metric("Spécificité", f"{spec * 100:.1f} %")

    a, b, c = st.columns(3)
    with a:
        afficher(g.matrice_confusion(confusion_matrix(ev["y_test"], pred)))
    with b:
        afficher(g.roc(ev["fpr"], ev["tpr"], ev["auc"]))
    with c:
        afficher(g.courbe_seuils(m.courbe_seuils(ev["y_test"], ev["proba"]), seuil))

    st.info("Hypothèses de la LDA : variables gaussiennes dans chaque classe et matrices de covariance égales. "
            "Elles ne sont qu'approximativement vérifiées sur ces données.")


def onglet_comparaison(df):
    st.subheader("LDA, régression logistique ou forêt aléatoire ?")
    st.write("Validation croisée stratifiée à 5 plis : chaque modèle est testé 5 fois sur des données "
             "qu'il n'a pas vues pendant l'entraînement.")
    res = comparaison(df)
    metriques = ["Accuracy", "AUC", "Sensibilité"]
    gauche, droite = st.columns([1, 1.4])
    with gauche:
        st.dataframe(res[metriques].style.format("{:.3f}").highlight_max(axis=0, color="#14532D"))
    with droite:
        afficher(g.comparaison(res, metriques))
    st.caption("Barres d'erreur : écart-type sur les 5 plis. Si les écarts entre modèles restent dans cette "
               "marge, la différence n'est pas significative : la LDA, plus simple et interprétable, se défend bien.")


def onglet_groupes(df, X_pca, patient_pca):
    st.subheader("Groupes de patients (K-means sur le premier plan factoriel)")
    k = st.segmented_control("Nombre de groupes", [2, 3, 4, 5], default=3) or 3
    km = groupes(X_pca[:, :2], k)
    etiquettes = km.labels_
    groupe_patient = int(km.predict(patient_pca[:, :2])[0])

    gauche, droite = st.columns([1.3, 1])
    with gauche:
        afficher(g.groupes(X_pca, etiquettes, patient_pca))
    with droite:
        st.success(f"Le patient actuel appartient au **groupe {groupe_patient + 1}**.")
        st.dataframe(m.resume_groupes(df, etiquettes))
        st.caption("Pct_diabetiques : part de diabétiques dans le groupe. Comparer le profil moyen du groupe "
                   "du patient à celui des autres permet d'adapter le suivi.")


def onglet_seuils(df):
    st.subheader("Seuils d'alerte personnalisés par tranche d'âge")
    st.write("Dans chaque tranche d'âge, la variable est modélisée par une loi normale pour les sains et pour les "
             "diabétiques, avec des paramètres estimés par **maximum de vraisemblance**. Le seuil d'alerte est la "
             "valeur à partir de laquelle la probabilité d'être diabétique dépasse 50 %.")
    variable = st.segmented_control("Variable", ["Glucose", "IMC", "Pression"], default="Glucose") or "Glucose"
    tableau = m.seuils_mle(df, variable)
    gauche, droite = st.columns([1.4, 1])
    with gauche:
        st.dataframe(tableau, hide_index=True)
    with droite:
        afficher(g.seuils_par_age(tableau, variable))
    st.caption("Un même taux de glucose n'a pas la même signification à 25 et à 50 ans : "
               "le seuil s'adapte au profil du patient.")


def onglet_exploration(df, pca):
    st.subheader("Statistiques descriptives par classe")
    moyennes = df.groupby("Classe")[m.VARIABLES].mean().round(2).rename(index={0: "Sain", 1: "Diabétique"})
    st.dataframe(moyennes)

    a, b = st.columns(2)
    with a:
        afficher(g.correlations(df[m.VARIABLES].corr()))
    with b:
        afficher(g.eboulis(pca.explained_variance_ratio_ * 100))

    charges = pca.components_[:2].T * (pca.explained_variance_[:2] ** 0.5)
    afficher(g.cercle_correlations(charges, m.VARIABLES, pca.explained_variance_ratio_))


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="entete">
            <h1>🩺 Assistant d'aide à la décision : dépistage du diabète</h1>
            <p class = "auteur">Réalisation de : Ange Odilon </p>
            <p>Statistique appliquée / Compétences utilisées (cours L2 SA / ENSPD): LDA, ACP, K-means et estimation par maximum de vraisemblance.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.warning("Outil pédagogique réalisé dans un cadre universitaire. "
               "Il ne remplace en aucun cas un avis médical.", icon="⚠️")

    df = charger()
    y = df["Classe"]
    scaler, lda, pca, X_scaled = modele_final(df)
    X_pca = pca.transform(X_scaled)
    ev = evaluation(df)

    patient, seuil = barre_laterale(df)
    patient_scaled = scaler.transform(patient)
    patient_pca = pca.transform(patient_scaled)

    onglets = st.tabs(["🔮 Diagnostic", "📈 Performance", "⚖️ Comparaison", "👥 Groupes", "🎯 Seuils", "🔎 Exploration"])
    with onglets[0]:
        onglet_diagnostic(patient_scaled, lda, pca, X_pca, y, patient_pca, seuil, ev)
    with onglets[1]:
        onglet_performance(ev, seuil)
    with onglets[2]:
        onglet_comparaison(df)
    with onglets[3]:
        onglet_groupes(df, X_pca, patient_pca)
    with onglets[4]:
        onglet_seuils(df)
    with onglets[5]:
        onglet_exploration(df, pca)


try:
    main()
except Exception as e:
    st.error(f"Erreur : {e}")
