# Databricks notebook source
"""
Dashboard Dash + Endpoint /predict
- Carga: obesidad.csv y modelo.pkl
- Dashboard: análisis descriptivo (histogramas, pie charts), análisis prescriptivo (correlación, feature importance / SHAP)
- Endpoint REST: POST /predict (JSON -> prediction)
- Formulario dentro del dashboard para ingresar un nuevo registro y obtener la predicción
"""

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from flask import request, jsonify
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# 1) CONFIGURACIÓN INICIAL
# -----------------------------
CSV_PATH = "ObesityDataSet.csv"
PKL_PATH = "modelo_random_forest.pkl"

variables_modelo = [
    'Gender', 'Age', 'Height', 'Weight',
    'family_history_with_overweight', 'FAVC',
    'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
    'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
]

# -----------------------------
# 2) CARGAR DATOS Y MODELO
# -----------------------------
df_model = pd.read_csv(CSV_PATH)
model = joblib.load(PKL_PATH)

CLASS_MAP = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

# -----------------------------
# 3) APP DASH
# -----------------------------
app = Dash(__name__, title="Dashboard Obesidad")
server = app.server

# -----------------------------
# 4) OPCIONES FORMULARIO
# -----------------------------
categorical_options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Walking', 'Bike', 'Motorbike', 'Car', 'Public_Transportation']
}

numeric_ranges = {
    'Age': (14, 61),
    'Height': (1.3, 2.0),
    'Weight': (40, 150),
    'FCVC': (1, 3),
    'NCP': (1, 4),
    'CH2O': (1, 3),
    'FAF': (0, 3),
    'TUE': (0, 2)
}

# -----------------------------
# 5) LAYOUT PRINCIPAL
# -----------------------------
app.layout = html.Div([

    html.H1("Dashboard de Análisis y Predicción de Obesidad", className="titulo"),

    # -------- FORMULARIO --------
    html.Div([
        html.H2("Formulario de Predicción", className="subtitulo"),
        html.P("Completa los campos. Usa listas desplegables y revisa los rangos sugeridos.", className="texto-ayuda"),

        html.Div([
            html.Div([
                html.Label(var, className="etiqueta"),
                dcc.Dropdown(
                    options=[{"label": opt, "value": opt} for opt in opts],
                    id=f"in-{var}",
                    value=opts[0],
                    clearable=False,
                    className="input-dropdown"
                )
            ]) for var, opts in categorical_options.items()
        ], className="bloque-form"),

        html.Div([
            html.Div([
                html.Label(var, className="etiqueta"),
                dcc.Input(id=f"in-{var}", type="number",
                          value=(rng[0] + rng[1]) / 2, step=0.1,
                          className="input-num"),
                html.Small(f"Rango sugerido: {rng[0]} - {rng[1]}", className="rango")
            ]) for var, rng in numeric_ranges.items()
        ], className="bloque-form"),

        html.Button("Predecir", id="btn-predict", n_clicks=0, className="boton"),
        html.Div(id="output-prediction", className="resultado")
    ], className="card"),

    # -------- ANÁLISIS DESCRIPTIVO --------
    html.Div([
        html.H2("Análisis Descriptivo", className="subtitulo"),
        dcc.Dropdown(
            id="hist-column",
            options=[{"label": c, "value": c} for c in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']],
            value="Weight",
            clearable=False,
            className="dropdown-descriptivo"
        ),
        dcc.Graph(id="hist-graph", className="grafica"),
        dcc.Graph(id="pie-graph", className="grafica"),
    ], className="card"),

    # -------- CORRELACIONES Y PDP --------
    html.Div([
        html.H2("Correlaciones y Explicabilidad del Modelo", className="subtitulo"),
        html.P("Explora las relaciones entre variables y cómo influyen en el modelo.", className="texto-ayuda"),
        dcc.Graph(id="corr-heatmap", className="grafica-grande"),
        html.Div(id="pdp-plot", className="pdp-container")
    ], className="card-grande")
])

# -----------------------------
# 6) CALLBACKS
# -----------------------------
@app.callback(
    Output("output-prediction", "children"),
    Input("btn-predict", "n_clicks"),
    [State(f"in-{v}", "value") for v in variables_modelo]
)
def on_predict(n_clicks, *vals):
    if n_clicks == 0:
        return ""
    data = {v: val for v, val in zip(variables_modelo, vals)}
    df_new = pd.DataFrame([data])
    #preds = model.predict(df_new)
    #return f"Predicción: {preds[0]}"
    preds = model.predict(df_new)
    pred_class = CLASS_MAP[int(preds[0])]
    return f"Predicción: {pred_class}"


@app.callback(
    Output("hist-graph", "figure"),
    Output("pie-graph", "figure"),
    Input("hist-column", "value")
)
def update_descriptive(col):
    fig_hist = px.histogram(df_model, x=col, nbins=20, title=f"Distribución de {col}", template="plotly_white", color_discrete_sequence=["#6baed6"])
    fig_pie = px.pie(df_model, names="Gender", title="Distribución por Género", hole=0.3, color_discrete_sequence=px.colors.sequential.Blues)
    return fig_hist, fig_pie

@app.callback(
    Output("corr-heatmap", "figure"),
    Output("pdp-plot", "children"),
    Input("btn-predict", "n_clicks")
)
def update_corr_pdp(n):
    df_num = df_model.select_dtypes(include=[np.number])
    corr = df_num.corr()
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="Blues",
        title="Mapa de Correlaciones Numéricas",
        template="plotly_white"
    )
    fig_corr.update_layout(width=800, height=600, title_font_size=22)

    # Partial Dependence Plot (multiclase: usa target=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        PartialDependenceDisplay.from_estimator(model, df_num, ['Age'], target=0, ax=ax)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        pdp_html = html.Img(src=f"data:image/png;base64,{img_b64}",
                            style={"width": "70%", "display": "block", "margin": "auto", "borderRadius": "10px"})
    except Exception as e:
        pdp_html = html.P(f"No se pudo generar PDP: {e}", style={"color": "red", "textAlign": "center"})

    return fig_corr, pdp_html

# -----------------------------
# 7) ENDPOINT REST
# -----------------------------
@server.route("/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        if isinstance(payload, dict) and "records" in payload:
            data = pd.DataFrame(payload["records"])
        elif isinstance(payload, dict):
            data = pd.DataFrame([payload])
        else:
            return jsonify({"error": "Formato JSON inválido"}), 400
        preds = model.predict(data)
        mapped = [CLASS_MAP[int(p)] for p in preds]
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 8) RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
