"""Graphiques interactifs Plotly de l'application.

Chaque fonction reçoit des données déjà calculées (par `modeles.py`) et
retourne une figure Plotly. Aucun code Streamlit ici.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

SAIN = "#38BDF8"
MALADE = "#F43F5E"
ACCENT = "#14B8A6"
PATIENT = "#FACC15"
GRILLE = "rgba(148,163,184,0.2)"


def _style(fig, hauteur=380):
    """Apparence commune : fond transparent, marges réduites, légende en bas."""
    fig.update_layout(
        height=hauteur,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor=GRILLE, zeroline=False)
    fig.update_yaxes(gridcolor=GRILLE, zeroline=False)
    return fig


def jauge(proba_pct, seuil):
    """Jauge de risque : vert / orange / rouge selon le seuil d'alerte."""
    s = seuil * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            number={"suffix": " %", "valueformat": ".1f"},
            title={"text": "Probabilité estimée de diabète"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E2E8F0", "thickness": 0.25},
                "steps": [
                    {"range": [0, s / 2], "color": "rgba(34,197,94,0.55)"},
                    {"range": [s / 2, s], "color": "rgba(245,158,11,0.55)"},
                    {"range": [s, 100], "color": "rgba(239,68,68,0.55)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": s},
            },
        )
    )
    return _style(fig, 280)


def contributions(contrib):
    """Contribution de chaque variable au score du patient (série triée)."""
    couleurs = [MALADE if v > 0 else SAIN for v in contrib.values]
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=couleurs))
    fig.update_layout(
        title="Pourquoi ce résultat ?",
        xaxis_title="← vers « sain »  |  vers « diabétique » →",
    )
    return _style(fig, 340)


def plan_factoriel(X_pca, y, patient_pca, ratios):
    """Nuage des individus sur le premier plan factoriel, avec le patient."""
    fig = go.Figure()
    for classe, nom, couleur in [(0, "Sain", SAIN), (1, "Diabétique", MALADE)]:
        masque = np.asarray(y == classe)
        fig.add_trace(
            go.Scatter(
                x=X_pca[masque, 0], y=X_pca[masque, 1], mode="markers", name=nom,
                marker=dict(color=couleur, size=6, opacity=0.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[patient_pca[0][0]], y=[patient_pca[0][1]], mode="markers", name="Patient actuel",
            marker=dict(color=PATIENT, size=20, symbol="star", line=dict(color="black", width=1.5)),
        )
    )
    fig.update_layout(
        title="Position du patient (ACP)",
        xaxis_title=f"Axe 1 ({ratios[0] * 100:.1f} %)",
        yaxis_title=f"Axe 2 ({ratios[1] * 100:.1f} %)",
    )
    return _style(fig, 420)


def matrice_confusion(cm):
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Teal",
        x=["Sain", "Diabétique"], y=["Sain", "Diabétique"],
        labels=dict(x="Prédit", y="Réel"),
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(title="Matrice de confusion")
    return _style(fig, 340)


def roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"LDA (AUC = {auc:.3f})",
                             line=dict(color=ACCENT, width=3), fill="tozeroy",
                             fillcolor="rgba(20,184,166,0.15)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Courbe ROC", xaxis_title="Faux positifs", yaxis_title="Vrais positifs")
    return _style(fig, 340)


def courbe_seuils(courbe, seuil):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=courbe["Seuil"], y=courbe["Sensibilité"], mode="lines",
                             name="Sensibilité", line=dict(color=MALADE, width=3)))
    fig.add_trace(go.Scatter(x=courbe["Seuil"], y=courbe["Spécificité"], mode="lines",
                             name="Spécificité", line=dict(color=SAIN, width=3)))
    fig.add_vline(x=seuil, line_dash="dash", line_color="white")
    fig.update_layout(title="Compromis sensibilité / spécificité", xaxis_title="Seuil d'alerte")
    return _style(fig, 340)


def comparaison(res, metriques):
    """Barres groupées avec écart-type de la validation croisée."""
    couleurs = [ACCENT, SAIN, MALADE]
    fig = go.Figure()
    for met, couleur in zip(metriques, couleurs):
        fig.add_trace(
            go.Bar(
                x=list(res.index), y=res[met], name=met, marker_color=couleur,
                error_y=dict(type="data", array=res[f"{met} (écart-type)"].to_numpy()),
            )
        )
    fig.update_layout(barmode="group", title="Comparaison des modèles (validation croisée)")
    fig.update_yaxes(range=[0, 1])
    return _style(fig, 400)


def groupes(X_pca, etiquettes, patient_pca):
    fig = go.Figure()
    for g in np.unique(etiquettes):
        masque = etiquettes == g
        fig.add_trace(
            go.Scatter(x=X_pca[masque, 0], y=X_pca[masque, 1], mode="markers", name=f"Groupe {g + 1}",
                       marker=dict(size=6, opacity=0.55))
        )
    fig.add_trace(
        go.Scatter(
            x=[patient_pca[0][0]], y=[patient_pca[0][1]], mode="markers", name="Patient actuel",
            marker=dict(color=PATIENT, size=20, symbol="star", line=dict(color="black", width=1.5)),
        )
    )
    fig.update_layout(title="Groupes de patients", xaxis_title="Axe 1", yaxis_title="Axe 2")
    return _style(fig, 440)


def seuils_par_age(tableau, variable):
    fig = go.Figure(go.Bar(x=tableau["Tranche d'âge"], y=tableau["Seuil d'alerte"], marker_color=ACCENT,
                           text=tableau["Seuil d'alerte"], textposition="outside"))
    fig.update_layout(title=f"Seuil d'alerte de {variable} par tranche d'âge")
    return _style(fig, 320)


def correlations(corr):
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Matrice de corrélation")
    return _style(fig, 440)


def eboulis(ratios_pct):
    axes = list(range(1, len(ratios_pct) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=axes, y=ratios_pct, name="Variance expliquée", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=axes, y=np.cumsum(ratios_pct), mode="lines+markers", name="Cumul",
                             line=dict(color=MALADE, width=3)))
    fig.update_layout(title="Éboulis des valeurs propres", xaxis_title="Composante",
                      yaxis_title="% de variance expliquée")
    return _style(fig, 400)


def cercle_correlations(charges, variables, ratios):
    fig = go.Figure()
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="gray", dash="dot"))
    for i, var in enumerate(variables):
        fig.add_trace(
            go.Scatter(
                x=[0, charges[i, 0]], y=[0, charges[i, 1]], mode="lines+text", text=["", var],
                textposition="top center", line=dict(color=MALADE, width=2), showlegend=False,
            )
        )
    fig.update_layout(
        title="Cercle des corrélations",
        xaxis=dict(range=[-1.2, 1.2], title=f"Axe 1 ({ratios[0] * 100:.1f} %)"),
        yaxis=dict(range=[-1.2, 1.2], title=f"Axe 2 ({ratios[1] * 100:.1f} %)", scaleanchor="x"),
    )
    return _style(fig, 480)
