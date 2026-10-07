# 🛒 Walmart — Prévision des ventes hebdomadaires (SARIMAX)

Pipeline de séries temporelles de bout en bout sur 45 magasins Walmart (143 semaines) :
EDA → stationnarité (ADF + KPSS) → ACF/PACF → baselines → SARIMAX avec variables exogènes → walk-forward → prévision à 12 semaines → application Streamlit.

🔗 **Application** :(https://walmart-sales-forecasting-hf6gdgmk4rtpvwqj6r6tj9.streamlit.app/)

![Prévision 12 semaines](assets/forecast_12_weeks.png)

## Résultats (test set = 29 dernières semaines)
| Modèle | MAPE |
|---|---|
| Naïf (dernière valeur) | 14,73 % |
| Naïf saisonnier (même semaine N-1) | 2,65 % |
| **SARIMAX(2,0,2)(1,0,0,52) + exogènes** | **2,12 %** |

> ⚠️ Le test (avril → octobre 2012) ne contient pas Noël. En walk-forward (4 plis × 13 semaines), l'erreur monte à ~10,5 % sur la période des fêtes quand une seule saison a été vue à l'entraînement : la limite est la profondeur d'historique (2,7 ans).

## Structure
```
├── Walmart_ST_Pipeline_Complet.ipynb   # notebook exécuté (résultats visibles)
├── app.py                              # application Streamlit
├── Walmart_Sales.csv                   # données (45 magasins × 143 semaines)
├── models/metrics.json                 # métriques de référence
├── assets/                             # visuels
├── requirements.txt                    # dépendances de l'app
└── requirements-notebook.txt           # + pmdarima pour relancer le notebook
```

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Notebook : `pip install -r requirements-notebook.txt` puis `jupyter notebook` → *Run All*.

## Application
Vue d'ensemble · analyse de la série · évaluation vs baselines · prévision à horizon libre avec IC 95 % · scénarios *what-if* (carburant, chômage, météo) · export CSV · upload de votre propre CSV.

## Limites & pistes
Modèle par magasin (hiérarchique) · XGBoost/LightGBM avec lags · promotions (MarkDown) · exogènes futures supposées (calendrier connu, météo N-1, autres = dernière valeur).

*Stack : Python · statsmodels · pmdarima · scikit-learn · Plotly · Streamlit.*
