import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
ARQ_LOG = BASE / "registro_previsoes.csv"

st.set_page_config(page_title="Preço de diamantes", page_icon="💎")


@st.cache_resource
def carregar():
    modelo = joblib.load(BASE / "modelos" / "modelo_publicado.joblib")
    info = json.load(open(BASE / "modelos" / "info_modelo.json", encoding="utf-8"))
    return modelo, info


modelo, info = carregar()
num = info["colunas_numericas"]
cat = info["colunas_categoricas"]

st.title("💎 Previsão do preço de diamantes")
st.caption(
    f"Modelo do experimento {info['experimento']}: {info['algoritmo']} ({info['o_que_mudou']}). "
    f"Erro médio no teste (MAE): US$ {info['metricas']['mae']:.0f}"
)

st.subheader("Características do diamante")
entradas = {}
c1, c2 = st.columns(2)
with c1:
    entradas["carat"] = st.number_input("carat (peso)", min_value=num["carat"][0], max_value=num["carat"][1] * 1.5,
                                        value=num["carat"][2], step=0.01)
    entradas["cut"] = st.selectbox("cut (lapidação, pior → melhor)", cat["cut"], index=len(cat["cut"]) - 1)
    entradas["color"] = st.selectbox("color (cor, J pior → D melhor)", cat["color"], index=3)
    entradas["clarity"] = st.selectbox("clarity (pureza, I1 pior → IF melhor)", cat["clarity"], index=3)
with c2:
    entradas["depth"] = st.number_input("depth (profundidade %)", min_value=num["depth"][0], max_value=num["depth"][1],
                                        value=num["depth"][2], step=0.1)
    entradas["table"] = st.number_input("table (mesa %)", min_value=num["table"][0], max_value=num["table"][1],
                                        value=num["table"][2], step=0.1)
    entradas["x"] = st.number_input("x (comprimento, mm)", min_value=num["x"][0], max_value=num["x"][1] * 1.5,
                                    value=num["x"][2], step=0.01)
    entradas["y"] = st.number_input("y (largura, mm)", min_value=num["y"][0], max_value=num["y"][1] * 1.5,
                                    value=num["y"][2], step=0.01)
    entradas["z"] = st.number_input("z (altura, mm)", min_value=num["z"][0], max_value=num["z"][1] * 1.5,
                                    value=num["z"][2], step=0.01)

if st.button("Prever preço", type="primary"):
    X_novo = pd.DataFrame([entradas])
    preco = float(modelo.predict(X_novo)[0])
    st.success(f"Preço previsto: **US$ {preco:,.2f}**")

    registro = pd.DataFrame([{
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entradas": json.dumps(entradas, ensure_ascii=False),
        "resultado": f"US$ {preco:.2f}",
        "experimento": info["experimento"],
    }])
    registro.to_csv(ARQ_LOG, mode="a", header=not ARQ_LOG.exists(), index=False, encoding="utf-8")

st.divider()
st.subheader("Registro de previsões")
if ARQ_LOG.exists():
    st.dataframe(pd.read_csv(ARQ_LOG).iloc[::-1])
else:
    st.info("Nenhuma previsão registrada ainda.")
