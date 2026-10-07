"""
🛒 Walmart Sales Forecasting — application Streamlit
Prévision des ventes hebdomadaires avec SARIMAX (variables exogènes) + baselines + scénarios.

Lancer :  streamlit run app.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", InterpolationWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & STYLE
# ──────────────────────────────────────────────────────────────────────────────
BLUE, YELLOW, RED, GREY, GREEN = "#0071CE", "#FFC220", "#E5383B", "#6B7280", "#2A9D8F"
DATA_PATH = os.path.join(os.path.dirname(__file__), "Walmart_Sales.csv")
EXOG_ALL = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]
PRESETS = {
    "SARIMAX (2,0,2)(1,0,0,52) — sélection auto_arima": ((2, 0, 2), (1, 0, 0, 52)),
    "SARIMAX (1,0,1)(1,0,0,52) — plus simple": ((1, 0, 1), (1, 0, 0, 52)),
    "SARIMAX (1,0,0)(1,0,0,52) — minimal": ((1, 0, 0), (1, 0, 0, 52)),
}
# Semaines "fériées" Walmart (Super Bowl, Labor Day, Thanksgiving, Noël) — calendrier connu à l'avance
HOLIDAY_WEEKS = pd.to_datetime([
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
    "2013-02-08", "2013-09-06", "2013-11-29", "2013-12-27",
    "2014-02-07", "2014-09-05", "2014-11-28", "2014-12-26",
])

st.set_page_config(page_title="Walmart Sales Forecasting", page_icon="🛒", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.6rem;}}
      .hero {{background: linear-gradient(120deg,{BLUE} 0%,#004C91 100%); padding: 1.4rem 1.8rem;
              border-radius: 14px; color: white; margin-bottom: 1.1rem;}}
      .hero h1 {{margin: 0; font-size: 1.9rem; color: white;}}
      .hero p {{margin: .3rem 0 0 0; opacity: .92;}}
      .hero .tag {{display:inline-block; background:{YELLOW}; color:#1b1b1b; padding:2px 10px;
                   border-radius:999px; font-size:.78rem; font-weight:700; margin-right:6px;}}
      div[data-testid="stMetric"] {{background:#F5F8FC; border:1px solid #E3EAF4; padding:.7rem .9rem;
                                    border-radius:12px;}}
    </style>
    <div class="hero">
      <h1>🛒 Walmart — Prévision des ventes hebdomadaires</h1>
      <p><span class="tag">SARIMAX</span><span class="tag">Séries temporelles</span>
         <span class="tag">45 magasins</span> Explorez les données, évaluez le modèle face aux baselines
         et simulez des scénarios.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def money(x, unit="M"):
    return f"{x / 1e6:,.1f} M$" if unit == "M" else f"{x:,.0f} $"


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


# ──────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv(source):
    raw = pd.read_csv(source)
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    return raw.dropna(subset=["Date"])


@st.cache_data(show_spinner=False)
def build_series(raw: pd.DataFrame, store):
    d = raw if store == "Tous les magasins" else raw[raw["Store"] == store]
    exog_cols = [c for c in EXOG_ALL if c in d.columns]
    sales = d.groupby("Date")["Weekly_Sales"].sum().rename("Weekly_Sales")
    df = pd.concat([sales, d.groupby("Date")[exog_cols].mean()], axis=1)
    day = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(pd.Series(df.index.dayofweek).mode()[0])]
    df = df.asfreq(f"W-{day}").interpolate().ffill().bfill()
    return df, exog_cols


with st.sidebar:
    st.header("⚙️ Paramètres")
    up = st.file_uploader("Votre fichier Walmart_Sales.csv (optionnel)", type="csv")
    try:
        raw = read_csv(up if up is not None else DATA_PATH)
        assert {"Store", "Date", "Weekly_Sales"} <= set(raw.columns)
    except Exception:
        st.error("Fichier invalide : colonnes requises `Store, Date, Weekly_Sales`.")
        st.stop()

    stores = ["Tous les magasins"] + sorted(raw["Store"].unique().tolist())
    store = st.selectbox("Périmètre", stores, help="Somme des 45 magasins ou un magasin précis")
    df, exog_cols = build_series(raw, store)

    st.markdown("**Modèle**")
    preset = st.selectbox("Spécification", list(PRESETS))
    order, sorder = PRESETS[preset]
    use_exog = st.toggle("Utiliser les variables exogènes", value=True,
                         help="Température, carburant, CPI, chômage, jours fériés")
    n_test = st.slider("Semaines de test", 13, 40, 29,
                       help="Les dernières semaines, jamais vues à l'entraînement")
    st.caption(f"{len(df)} semaines · {df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y}")

if len(df) < 110:
    st.warning("Moins de 110 semaines : la saisonnalité annuelle (m=52) sera difficile à estimer.")

y = df["Weekly_Sales"]
X = df[exog_cols] if (use_exog and exog_cols) else None


# ──────────────────────────────────────────────────────────────────────────────
# MODÈLES
# ──────────────────────────────────────────────────────────────────────────────
def fit_sarimax(y_, X_, order_, sorder_):
    return SARIMAX(y_, exog=X_, order=order_, seasonal_order=sorder_,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


@st.cache_data(show_spinner="Entraînement du modèle…")
def evaluate(_df, key, exog_used, order_, sorder_, n_test_):
    cols = list(exog_used)
    tr, te = _df.iloc[:-n_test_], _df.iloc[-n_test_:]
    res = fit_sarimax(tr["Weekly_Sales"], tr[cols] if cols else None, order_, sorder_)
    fc = res.get_forecast(len(te), exog=te[cols] if cols else None)
    pred, ci = fc.predicted_mean, fc.conf_int()
    naive = pd.Series(tr["Weekly_Sales"].iloc[-1], index=te.index)
    snaive = _df["Weekly_Sales"].shift(52).reindex(te.index) if n_test_ <= 52 else naive * np.nan
    yt = te["Weekly_Sales"]
    rows = []
    for name, p in [("Naïf (dernière valeur)", naive), ("Naïf saisonnier (N-1)", snaive),
                    ("SARIMAX", pred)]:
        if p.isna().any():
            continue
        rows.append({"Modèle": name, "MAE": mean_absolute_error(yt, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yt, p))), "MAPE (%)": mape(yt, p)})
    lb = acorr_ljungbox(res.resid.iloc[52:].dropna(), lags=[10], return_df=True)["lb_pvalue"].iloc[0] \
        if len(res.resid) > 70 else np.nan
    return {"table": pd.DataFrame(rows), "pred": pred, "ci": ci, "naive": naive, "snaive": snaive,
            "train": tr["Weekly_Sales"], "test": yt, "aic": res.aic, "lb_p": lb,
            "coef": res.params, "pvals": res.pvalues}


@st.cache_data(show_spinner="Validation croisée walk-forward…")
def walk_forward(_df, key, exog_used, order_, sorder_):
    cols = list(exog_used)
    rows = []
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=4, test_size=13).split(_df), 1):
        tr, te = _df.iloc[a], _df.iloc[b]
        res = fit_sarimax(tr["Weekly_Sales"], tr[cols] if cols else None, order_, sorder_)
        p = res.get_forecast(len(te), exog=te[cols] if cols else None).predicted_mean
        s = _df["Weekly_Sales"].shift(52).reindex(te.index)
        rows.append({"Pli": k, "Train (sem.)": len(tr),
                     "Période test": f"{te.index[0]:%d/%m/%y} → {te.index[-1]:%d/%m/%y}",
                     "MAPE SARIMAX (%)": mape(te["Weekly_Sales"], p),
                     "MAPE Saisonnier (%)": mape(te["Weekly_Sales"], s)})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Entraînement sur 100 % des données…")
def fit_full(_df, key, exog_used, order_, sorder_):
    cols = list(exog_used)
    return fit_sarimax(_df["Weekly_Sales"], _df[cols] if cols else None, order_, sorder_)


def future_exog(df_, cols, steps, d_fuel=0.0, d_unemp=0.0, d_temp=0.0, extra_holidays=()):
    idx = pd.date_range(df_.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq=df_.index.freq)
    fx = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for c in cols:
        if c == "Holiday_Flag":
            fx[c] = (idx.isin(HOLIDAY_WEEKS) | idx.isin(pd.to_datetime(list(extra_holidays)))).astype(float)
        elif c == "Temperature":
            fx[c] = df_["Temperature"].reindex(idx - pd.Timedelta(weeks=52)).values + d_temp
        elif c == "Fuel_Price":
            fx[c] = df_[c].iloc[-1] * (1 + d_fuel / 100)
        elif c == "Unemployment":
            fx[c] = df_[c].iloc[-1] + d_unemp
        else:
            fx[c] = df_[c].iloc[-1]
    return fx.ffill().bfill()


key = f"{store}|{len(df)}|{df.index[-1]}|{order}|{sorder}|{use_exog}|{n_test}"
exog_used = tuple(exog_cols) if X is not None else ()

# ──────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Vue d'ensemble", "🔬 Analyse de la série", "🤖 Modèle & évaluation", "🔮 Prévision & scénarios", "ℹ️ À propos"]
)

# ── 1. Vue d'ensemble ────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventes cumulées", f"{y.sum() / 1e9:,.2f} Md$")
    c2.metric("Moyenne hebdo", money(y.mean()))
    c3.metric("Pic hebdo", money(y.max()), f"{y.idxmax():%d/%m/%Y}", delta_color="off")
    c4.metric("Coef. de variation", f"{y.std() / y.mean() * 100:.1f} %")

    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y, name="Ventes hebdo", line=dict(color=BLUE, width=1.6), opacity=.6)
    for w, col in [(4, YELLOW), (12, RED)]:
        fig.add_scatter(x=y.index, y=y.rolling(w).mean(), name=f"Moyenne mobile {w} sem.",
                        line=dict(color=col, width=2.4))
    hol = df.index[df["Holiday_Flag"] >= 0.5] if "Holiday_Flag" in df else []
    fig.add_scatter(x=hol, y=y.loc[hol], mode="markers", name="Semaine fériée",
                    marker=dict(color=RED, size=8, symbol="diamond"))
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      title="Ventes hebdomadaires", yaxis_title="$", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    colA, colB = st.columns(2)
    wk = y.groupby(df.index.isocalendar().week.astype(int).values).mean()
    figw = go.Figure(go.Bar(x=wk.index, y=wk.values, marker_color=[YELLOW if v > wk.mean() * 1.1 else BLUE for v in wk.values]))
    figw.update_layout(template="plotly_white", height=330, title="Ventes moyennes par semaine de l'année",
                       xaxis_title="Semaine ISO", yaxis_title="$")
    colA.plotly_chart(figw, width="stretch")
    if exog_cols:
        corr = df[["Weekly_Sales"] + exog_cols].corr()["Weekly_Sales"].drop("Weekly_Sales").sort_values()
        figc = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                                marker_color=[RED if v < 0 else GREEN for v in corr.values]))
        figc.update_layout(template="plotly_white", height=330, title="Corrélation avec les ventes",
                           xaxis=dict(range=[-1, 1]))
        colB.plotly_chart(figc, width="stretch")

    if store == "Tous les magasins":
        top = raw.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False).head(10)
        figt = go.Figure(go.Bar(x=[f"Magasin {s}" for s in top.index], y=top.values, marker_color=BLUE))
        figt.update_layout(template="plotly_white", height=320, title="Top 10 magasins — vente hebdo moyenne",
                           yaxis_title="$")
        st.plotly_chart(figt, width="stretch")

# ── 2. Analyse ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Décomposition tendance / saisonnalité / résidu")
    if len(y) >= 104:
        dec = seasonal_decompose(y, model="additive", period=52)
        figd = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
                             subplot_titles=("Observée", "Tendance", "Saisonnalité (52 sem.)", "Résidu"))
        for i, (s, col) in enumerate([(dec.observed, BLUE), (dec.trend, YELLOW), (dec.seasonal, GREEN), (dec.resid, RED)], 1):
            figd.add_scatter(x=s.index, y=s, line=dict(color=col), row=i, col=1, showlegend=False)
        figd.update_layout(template="plotly_white", height=640)
        st.plotly_chart(figd, width="stretch")
        res_ = dec.resid.dropna()
        fs = max(0, 1 - np.var(res_) / np.var((dec.seasonal + dec.resid).dropna()))
        ft = max(0, 1 - np.var(res_) / np.var((dec.trend + dec.resid).dropna()))
        a, b = st.columns(2)
        a.metric("Force de la saisonnalité", f"{fs:.2f}", help="0 = absente · 1 = très forte")
        b.metric("Force de la tendance", f"{ft:.2f}", help="0 = absente · 1 = très forte")
    else:
        st.info("Il faut au moins 104 semaines pour décomposer avec une période de 52.")

    st.subheader("Stationnarité (ADF + KPSS)")
    adf_p = adfuller(y, autolag="AIC")[1]
    kpss_p = kpss(y, regression="c", nlags="auto")[1]
    c1, c2, c3 = st.columns(3)
    c1.metric("ADF p-value", f"{adf_p:.2e}", "stationnaire" if adf_p < .05 else "non stationnaire",
              delta_color="off", help="H₀ : racine unitaire. p < 0,05 → stationnaire")
    c2.metric("KPSS p-value", f"{kpss_p:.3f}", "stationnaire" if kpss_p > .05 else "non stationnaire",
              delta_color="off", help="H₀ : stationnaire. p > 0,05 → stationnaire")
    verdict = "✅ Série stationnaire → **d = 0**" if (adf_p < .05 and kpss_p > .05) else "⚠️ Non stationnaire → **d = 1** recommandé"
    c3.markdown(f"<br>{verdict}", unsafe_allow_html=True)

    st.subheader("ACF / PACF")
    n = len(y)
    a_vals = acf(y, nlags=min(80, n - 2))
    p_vals = pacf(y, nlags=min(60, n // 2 - 1), method="ols")
    band = 1.96 / np.sqrt(n)
    figa = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    for j, v in enumerate([a_vals, p_vals], 1):
        figa.add_bar(x=list(range(len(v))), y=v, marker_color=BLUE, showlegend=False, row=1, col=j)
        figa.add_hline(y=band, line_dash="dot", line_color=RED, row=1, col=j)
        figa.add_hline(y=-band, line_dash="dot", line_color=RED, row=1, col=j)
    figa.update_layout(template="plotly_white", height=340)
    st.plotly_chart(figa, width="stretch")
    st.caption("Un pic marqué autour du lag 52 confirme la saisonnalité annuelle (m = 52).")

# ── 3. Modèle & évaluation ───────────────────────────────────────────────────
with tab3:
    ev = evaluate(df, key, exog_used, order, sorder, n_test)
    tbl = ev["table"].set_index("Modèle")
    sar, best_base = tbl.loc["SARIMAX", "MAPE (%)"], tbl.drop("SARIMAX")["MAPE (%)"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE SARIMAX", f"{sar:.2f} %")
    c2.metric("Meilleure baseline", f"{best_base:.2f} %")
    c3.metric("Gain vs baseline", f"{best_base - sar:+.2f} pt", f"{(best_base - sar) / best_base * 100:+.0f} % d'erreur",
              delta_color="normal")
    c4.metric("AIC", f"{ev['aic']:,.0f}")

    st.dataframe(
        tbl.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}"})
           .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#DFF3E4"),
        width="stretch",
    )

    figm = go.Figure()
    figm.add_scatter(x=ev["train"].index[-60:], y=ev["train"].iloc[-60:], name="Train (60 dernières sem.)",
                     line=dict(color=BLUE, width=1.4), opacity=.55)
    figm.add_scatter(x=ev["test"].index, y=ev["test"], name="Réel (test)", line=dict(color="black", width=2.4))
    figm.add_scatter(x=ev["ci"].index, y=ev["ci"].iloc[:, 1], line=dict(width=0), showlegend=False, hoverinfo="skip")
    figm.add_scatter(x=ev["ci"].index, y=ev["ci"].iloc[:, 0], fill="tonexty", fillcolor="rgba(229,56,59,.15)",
                     line=dict(width=0), name="IC 95 %", hoverinfo="skip")
    figm.add_scatter(x=ev["pred"].index, y=ev["pred"], name=f"SARIMAX ({sar:.1f} %)", line=dict(color=RED, width=2.6))
    if not ev["snaive"].isna().any():
        figm.add_scatter(x=ev["snaive"].index, y=ev["snaive"], name="Naïf saisonnier",
                         line=dict(color=GREEN, width=1.8, dash="dash"))
    figm.update_layout(template="plotly_white", height=430, hovermode="x unified", yaxis_title="$",
                       title="Réel vs prédit sur la période de test", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figm, width="stretch")

    in_test = ev["test"].index
    if not ((in_test.month >= 11) | (in_test.month == 1)).any():
        st.info("ℹ️ La période de test ne contient **ni Thanksgiving ni Noël** : l'erreur affichée est plus "
                "favorable que sur la période des fêtes. La validation walk-forward ci-dessous inclut cette période.")

    colx, coly = st.columns(2)
    with colx:
        st.markdown("**Diagnostic des résidus**")
        if not np.isnan(ev["lb_p"]):
            st.write(f"Ljung-Box (10 lags) p = **{ev['lb_p']:.3f}** → "
                     f"{'✅ résidus ≈ bruit blanc' if ev['lb_p'] > .05 else '⚠️ autocorrélation résiduelle'}")
    with coly:
        if exog_used:
            st.markdown("**Effet des variables exogènes**")
            co = pd.DataFrame({"Coefficient": ev["coef"][list(exog_used)], "p-value": ev["pvals"][list(exog_used)]})
            co["Signif. 5 %"] = np.where(co["p-value"] < .05, "✅", "—")
            st.dataframe(co.style.format({"Coefficient": "{:,.0f}", "p-value": "{:.3f}"}), width="stretch")

    with st.expander("🔄 Validation croisée walk-forward (4 plis × 13 semaines) — ~20 s"):
        if st.button("Lancer la validation croisée"):
            cv = walk_forward(df, key, exog_used, order, sorder)
            st.dataframe(cv.set_index("Pli").style.format({"MAPE SARIMAX (%)": "{:.2f}", "MAPE Saisonnier (%)": "{:.2f}"}),
                         width="stretch")
            figcv = go.Figure()
            figcv.add_bar(x=cv["Pli"], y=cv["MAPE SARIMAX (%)"], name="SARIMAX", marker_color=RED)
            figcv.add_bar(x=cv["Pli"], y=cv["MAPE Saisonnier (%)"], name="Naïf saisonnier", marker_color=GREEN)
            figcv.update_layout(template="plotly_white", barmode="group", height=300, xaxis_title="Pli", yaxis_title="MAPE (%)")
            st.plotly_chart(figcv, width="stretch")
            st.caption("Le modèle est plus fragile quand il n'a vu qu'une seule saison de fêtes : "
                       "la limite est la profondeur d'historique, pas l'algorithme.")

# ── 4. Prévision & scénarios ─────────────────────────────────────────────────
with tab4:
    st.subheader("Prévision à horizon libre")
    c1, c2 = st.columns([1, 2])
    steps = c1.slider("Horizon (semaines)", 4, 26, 12)
    c2.caption("Variables exogènes futures : **jours fériés** = calendrier connu · **température** = même semaine N-1 · "
               "**carburant / CPI / chômage** = dernière valeur (+ ajustement ci-dessous).")

    with st.expander("🎛️ Scénarios « what-if »", expanded=True):
        s1, s2, s3 = st.columns(3)
        d_fuel = s1.slider("Prix du carburant (%)", -30, 30, 0, 5)
        d_unemp = s2.slider("Chômage (points)", -2.0, 2.0, 0.0, 0.25)
        d_temp = s3.slider("Température (°F)", -10, 10, 0, 1)

    model = fit_full(df, key, exog_used, order, sorder)
    if exog_used:
        fx_base = future_exog(df, list(exog_used), steps)
        fx_scn = future_exog(df, list(exog_used), steps, d_fuel, d_unemp, d_temp)
    else:
        fx_base = fx_scn = None
    f_base = model.get_forecast(steps, exog=fx_base)
    pred_b, ci_b = f_base.predicted_mean, f_base.conf_int()
    scenario_on = exog_used and (d_fuel or d_unemp or d_temp)
    pred_s = model.get_forecast(steps, exog=fx_scn).predicted_mean if scenario_on else None

    ly = y.reindex(pred_b.index - pd.Timedelta(weeks=52))
    figf = go.Figure()
    figf.add_scatter(x=y.index[-52:], y=y.iloc[-52:], name="Historique", line=dict(color=BLUE, width=1.8))
    figf.add_scatter(x=ci_b.index, y=ci_b.iloc[:, 1], line=dict(width=0), showlegend=False, hoverinfo="skip")
    figf.add_scatter(x=ci_b.index, y=ci_b.iloc[:, 0], fill="tonexty", fillcolor="rgba(229,56,59,.15)",
                     line=dict(width=0), name="IC 95 %", hoverinfo="skip")
    figf.add_scatter(x=pred_b.index, y=ly.values, name="Même période N-1", line=dict(color=GREY, dash="dot", width=1.8))
    figf.add_scatter(x=pred_b.index, y=pred_b, name="Prévision (base)", mode="lines+markers",
                     line=dict(color=RED, width=2.6), marker=dict(size=6))
    if pred_s is not None:
        figf.add_scatter(x=pred_s.index, y=pred_s, name="Scénario", mode="lines+markers",
                         line=dict(color=YELLOW, width=2.6), marker=dict(size=6, symbol="diamond"))
    figf.add_vline(x=y.index[-1], line_dash="dash", line_color=GREY)
    figf.update_layout(template="plotly_white", height=470, hovermode="x unified", yaxis_title="$",
                       title=f"Prévision — {steps} prochaines semaines ({store})", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figf, width="stretch")

    k1, k2, k3 = st.columns(3)
    k1.metric("Total prévu", money(pred_b.sum()))
    k2.metric("Semaine la plus forte", money(pred_b.max()), f"{pred_b.idxmax():%d/%m/%Y}", delta_color="off")
    if pred_s is not None:
        dlt = pred_s.sum() - pred_b.sum()
        k3.metric("Impact du scénario", money(pred_s.sum()), f"{dlt / 1e6:+,.1f} M$ ({dlt / pred_b.sum() * 100:+.1f} %)")
    else:
        k3.metric("Vs même période N-1", f"{(pred_b.sum() / ly.sum() - 1) * 100:+.1f} %" if ly.notna().all() else "n/a")

    out = pd.DataFrame({"Semaine": pred_b.index.date, "Prévision ($)": pred_b.round(0).values,
                        "IC inf ($)": ci_b.iloc[:, 0].round(0).values, "IC sup ($)": ci_b.iloc[:, 1].round(0).values})
    if pred_s is not None:
        out["Scénario ($)"] = pred_s.round(0).values
    st.dataframe(out.style.format({c: "{:,.0f}" for c in out.columns if "$" in c}), width="stretch", hide_index=True)
    st.download_button("⬇️ Télécharger la prévision (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="walmart_forecast.csv", mime="text/csv")

# ── 5. À propos ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        """
### Méthode
1. **Préparation** — agrégation multi-magasins (somme des ventes, moyenne des exogènes), fréquence hebdomadaire fixe, interpolation.
2. **Analyse** — décomposition, tests ADF + KPSS (→ *d*), ACF/PACF (→ saisonnalité *m = 52*).
3. **Baselines** — naïf (dernière valeur) et naïf saisonnier (même semaine N-1) : le plancher à battre.
4. **Modèle** — SARIMAX sélectionné par `auto_arima` (AIC), variables exogènes : température, carburant, CPI, chômage, jours fériés.
5. **Évaluation** — split chronologique + walk-forward ; MAE / RMSE / MAPE ; diagnostic des résidus.
6. **Prévision** — réentraînement sur 100 % des données, intervalle de confiance 95 %, scénarios *what-if*.

### Limites assumées
- ~2,7 ans d'historique seulement : la saison annuelle n'est observée que deux fois, d'où une erreur plus forte autour de Noël quand une seule saison a été vue.
- Sur le jeu de test, les exogènes réelles sont fournies au modèle ; en prévision elles sont **supposées** (voir l'onglet Prévision).
- La baseline saisonnière est très compétitive sur une série aussi régulière — le gain du SARIMAX est réel mais modeste.
- Pistes : modèle par magasin (hiérarchique), XGBoost/LightGBM avec lags, prise en compte des promotions (MarkDown).

*Stack : Python · statsmodels · pmdarima · scikit-learn · Plotly · Streamlit.*
"""
    )
