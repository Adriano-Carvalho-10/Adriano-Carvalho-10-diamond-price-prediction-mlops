import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
ARQ_LOG = BASE / "registro_previsoes.csv"

st.set_page_config(page_title="Preço de diamantes", page_icon="💎")


@st.cache_resource
def carregar():
    modelo = joblib.load(BASE / "modelos" / "modelo_publicado.joblib")
    info = json.load(open(BASE / "modelos" / "info_modelo.json", encoding="utf-8"))
    return modelo, info


@st.cache_data
def carregar_base():
    return pd.read_csv(BASE / "dados" / "diamonds.csv")


@st.cache_data
def carregar_experimentos():
    return pd.read_csv(BASE / "modelos" / "tabela_experimentos.csv")


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

st.divider()
with st.expander("Explorar a base de dados"):
    st.caption("Gráficos interativos com a base usada no treinamento.")
    dados = carregar_base()

    aba1, aba2, aba3 = st.tabs([
        "Preço x característica",
        "Preço médio por categoria",
        "Distribuição do preço",
    ])

    with aba1:
        coluna_x = st.selectbox("Característica numérica (eixo X)", list(num.keys()), key="explorar_x")
        cor = st.selectbox("Colorir por", ["Nenhuma"] + list(cat.keys()), key="explorar_cor")
        amostra = st.slider("Pontos exibidos", 1000, len(dados), min(5000, len(dados)), 1000,
                            key="explorar_amostra")
        dados_plot = dados if amostra >= len(dados) else dados.sample(n=amostra, random_state=42)
        cor_selecionada = None if cor == "Nenhuma" else cor
        hover = [c for c in ["carat", "cut", "color", "clarity"]
                 if c != coluna_x and c != cor_selecionada]
        fig = px.scatter(
            dados_plot,
            x=coluna_x,
            y="price",
            color=cor_selecionada,
            opacity=0.45,
            hover_data=hover,
            title=f"Preço x {coluna_x}",
            labels={"price": "Preço (US$)"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Exibindo {amostra:,} de {len(dados):,} diamantes.")

    with aba2:
        categoria = st.selectbox("Categoria", list(cat.keys()), key="explorar_cat")
        ordem = cat[categoria]
        media = (
            dados.groupby(categoria)["price"]
            .mean()
            .reindex(ordem)
            .reset_index()
        )
        media["texto"] = media["price"].map(lambda valor: f"US$ {valor:,.0f}")
        fig = px.bar(
            media,
            x=categoria,
            y="price",
            text="texto",
            category_orders={categoria: ordem},
            title=f"Preço médio por {categoria}",
            labels={"price": "Preço médio (US$)"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    with aba3:
        faixa = st.slider(
            "Faixa de carat",
            num["carat"][0],
            num["carat"][1],
            (num["carat"][0], num["carat"][1]),
            0.01,
            key="explorar_faixa",
        )
        filtrada = dados[dados["carat"].between(faixa[0], faixa[1])]
        fig = px.histogram(
            filtrada,
            x="price",
            nbins=80,
            marginal="box",
            title="Distribuição do preço",
            labels={"price": "Preço (US$)"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(filtrada):,} diamantes na faixa selecionada.")

st.divider()
with st.expander("Comparar experimentos"):
    st.caption("Os 7 experimentos foram avaliados no mesmo teste (20% das linhas).")
    experimentos = carregar_experimentos()

    m1, m2, m3 = st.columns(3)
    m1.metric("Melhor RMSE", f"US$ {experimentos['rmse'].min():,.1f}")
    m2.metric("Melhor MAE", f"US$ {experimentos['mae'].min():,.1f}")
    m3.metric("Melhor R²", f"{experimentos['r2'].max():.4f}")

    metrica = st.selectbox(
        "Métrica do gráfico",
        ["rmse", "mae", "r2"],
        format_func=lambda m: {
            "rmse": "RMSE (menor é melhor)",
            "mae": "MAE (menor é melhor)",
            "r2": "R² (maior é melhor)",
        }[m],
        key="comparar_metrica",
    )

    comparativo = experimentos.assign(
        Experimento=experimentos["numero"].map(lambda n: f"Exp. {n}"),
        Destaque=experimentos["numero"].eq(info["experimento"]).map(
            {True: "Publicado", False: "Testado"}
        ),
    )
    comparativo["texto"] = comparativo[metrica].map(
        lambda v: f"{v:.4f}" if metrica == "r2" else f"US$ {v:,.1f}"
    )
    comparativo = comparativo.sort_values(metrica, ascending=(metrica != "r2"))

    fig = px.bar(
        comparativo,
        x="Experimento",
        y=metrica,
        color="Destaque",
        text="texto",
        category_orders={"Destaque": ["Publicado", "Testado"]},
        color_discrete_map={"Publicado": "#2ca02c", "Testado": "#636efa"},
        hover_data=["algoritmo", "o_que_mudou", "rmse", "mae", "r2", "tempo_treino_s"],
        title=f"Comparação dos experimentos por {metrica.upper()}",
        labels={
            "rmse": "RMSE (US$)",
            "mae": "MAE (US$)",
            "r2": "R²",
            "Experimento": "",
            "Destaque": "Modelo",
        },
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    tabela = experimentos.assign(
        Status=experimentos["numero"].eq(info["experimento"]).map(
            {True: "Publicado", False: "Testado"}
        )
    )[[
        "numero",
        "algoritmo",
        "o_que_mudou",
        "rmse",
        "mae",
        "r2",
        "tempo_treino_s",
        "Status",
    ]].sort_values("numero")

    st.dataframe(
        tabela.style.format({
            "rmse": "US$ {:,.1f}",
            "mae": "US$ {:,.1f}",
            "r2": "{:.4f}",
            "tempo_treino_s": "{:.2f} s",
        }),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Modelo publicado: experimento {info['experimento']}.")
