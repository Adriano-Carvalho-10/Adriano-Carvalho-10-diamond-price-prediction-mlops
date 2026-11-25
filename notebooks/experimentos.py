# %% [markdown]
# # MLOps - 3ª Avaliação
# ## Tarefa: regressão - previsão do preço de diamantes (Diamonds)
# Base pública Diamonds (≈54 mil diamantes, a mesma do pacote ggplot2). A pergunta é: dadas as características de um diamante, qual o preço em dólares?

# %%
import json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

SEED = 42

# %% [markdown]
# ## 1. A base

# %%
df = pd.read_csv("../dados/diamonds.csv")
print("linhas e colunas:", df.shape)
df.head()

# %%
print(df.dtypes)
print("\nvalores ausentes:", df.isna().sum().sum())
print("linhas duplicadas:", df.duplicated().sum())
print("diamantes com x, y ou z igual a 0:", (df[["x", "y", "z"]] == 0).any(axis=1).sum())
print("y > 30 mm:", (df["y"] > 30).sum(), "| z > 30 mm:", (df["z"] > 30).sum())
df["price"].describe().round(1)

# %% [markdown]
# ## 2. Pré-processamento
# - **Duplicadas (146 linhas):** removidas, para a mesma linha não aparecer no treino e no teste.
# - **Dimensões zeradas (20 linhas) e dimensões absurdas (y ou z acima de 30 mm, 3 linhas):** são erros de cadastro (um diamante não tem 0 mm nem 58 mm), então saem.
# - **Colunas categóricas (cut, color, clarity):** One-Hot Encoding dentro do Pipeline.
# - **Colunas numéricas:** mediana para ausentes (não há nenhum, mas o pipeline fica seguro para o app) e padronização, que importa para a Regressão Linear e o KNN.
# - Colunas: `carat` (peso), `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z` (medidas em mm). Alvo: `price`.

# %%
antes = len(df)
df = df.drop_duplicates()
df = df[(df[["x", "y", "z"]] > 0).all(axis=1) & (df["y"] < 30) & (df["z"] < 30)]
print(f"linhas: {antes} -> {len(df)}")

y = df.pop("price")
X = df
NUM = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
CAT = [c for c in X.columns if c not in NUM]
print("numéricas:", NUM)
print("categóricas:", CAT)

# %% [markdown]
# ## 3. Conjunto de teste único
# 20% das linhas, separado uma vez só. Todos os experimentos são avaliados nele.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(X_tr.shape, X_te.shape)

def montar_pipeline(modelo, colunas, log_alvo=False):
    num = [c for c in colunas if c in NUM]
    cat = [c for c in colunas if c in CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    if log_alvo:
        modelo = TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("pre", pre), ("modelo", modelo)])

linhas, pipelines = [], {}

def rodar(num, nome_algoritmo, o_que_mudou, modelo, colunas=None, n_linhas=None, log_alvo=False, publicavel=True):
    colunas = colunas or list(X.columns)
    Xt, yt = X_tr[colunas], y_tr
    if n_linhas:
        Xt, yt = Xt.iloc[:n_linhas], yt.iloc[:n_linhas]
    t0 = time.time()
    pipe = montar_pipeline(modelo, colunas, log_alvo).fit(Xt, yt)
    tempo = time.time() - t0
    pred = pipe.predict(X_te[colunas])
    r = {
        "numero": num, "algoritmo": nome_algoritmo, "o_que_mudou": o_que_mudou,
        "rmse": round(mean_squared_error(y_te, pred) ** 0.5, 1),
        "mae": round(mean_absolute_error(y_te, pred), 1),
        "r2": round(r2_score(y_te, pred), 4),
        "tempo_treino_s": round(tempo, 2),
        "publicavel": publicavel,
    }
    linhas.append(r)
    pipelines[num] = pipe
    return r

# %% [markdown]
# ## 4. Experimentos
# Métrica principal: **RMSE em dólares** (erro típico, penaliza erros grandes). MAE e R² acompanham.
# Algoritmos: Regressão Linear, KNN, Random Forest e Gradient Boosting (HistGradientBoosting).

# %%
rodar(1, "Regressão Linear", "Baseline: todas as colunas, parâmetros padrão", LinearRegression())
rodar(2, "KNN", "Troca o algoritmo: 10 vizinhos", KNeighborsRegressor(n_neighbors=10))
rodar(3, "Random Forest", "Troca o algoritmo: 100 árvores (min_samples_leaf=5 para o arquivo ficar leve)",
      RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=-1))
rodar(4, "Gradient Boosting", "Troca o algoritmo: HistGradientBoosting",
      HistGradientBoostingRegressor(random_state=SEED))

# %% [markdown]
# ### Experimentos que mudam os dados
# No experimento 5, a coluna mais importante (por importância de permutação no Gradient Boosting) sai do treino.
# No experimento 6, o Random Forest treina com só metade das linhas. O teste continua o mesmo.
# O experimento 7 muda o alvo: o modelo aprende o logaritmo do preço e a previsão é convertida de volta.

# %%
imp = permutation_importance(pipelines[4], X_te, y_te, scoring="neg_root_mean_squared_error",
                             n_repeats=3, random_state=SEED, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print(imp.round(1))
MAIS_IMPORTANTE = imp.index[0]
print("\ncoluna removida no experimento 5:", MAIS_IMPORTANTE)

# %%
sem_col = [c for c in X.columns if c != MAIS_IMPORTANTE]
rodar(5, "Gradient Boosting", f"Dados: remove a coluna mais importante ({MAIS_IMPORTANTE})",
      HistGradientBoostingRegressor(random_state=SEED), colunas=sem_col, publicavel=False)
rodar(6, "Random Forest", f"Dados: treina com metade das linhas ({len(X_tr)//2} de {len(X_tr)})",
      RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
      n_linhas=len(X_tr)//2, publicavel=False)
rodar(7, "Gradient Boosting", "Dados: alvo em escala logarítmica (log do preço)",
      HistGradientBoostingRegressor(random_state=SEED), log_alvo=True)

# %% [markdown]
# ## 5. Tabela de experimentos

# %%
OBS = {
    1: "Ponto de partida: RMSE de US$ 1.122. O preço não cresce em linha reta com o peso, e a reta não captura isso.",
    2: "O KNN já derruba o RMSE para US$ 782, sem treino de verdade (0,09 s), mas depende das colunas estarem padronizadas.",
    3: "Menor RMSE e menor MAE da tabela. Custo: é o treino mais lento (cerca de 13 s) e o arquivo do modelo é o maior (cerca de 17 MB).",
    4: "Quase empata com o Random Forest (RMSE 542 contra 538) e treina em cerca de 1 s.",
    5: "Sem a coluna y o RMSE quase não muda (545 contra 542): x, z e carat trazem a mesma informação, porque as medidas são muito correlacionadas.",
    6: "Com metade das linhas o RMSE sobe de 538 para 574, cerca de 7% pior. Mais dados ajudam, mas pouco.",
    7: "Aprender o log do preço piorou o RMSE (557 contra 542) e melhorou o MAE (284 contra 290): erra menos nos diamantes baratos e mais nos caros.",
}
tabela = pd.DataFrame(linhas)
tabela["resultado"] = "RMSE US$ " + tabela["rmse"].astype(str) + " | MAE US$ " + tabela["mae"].astype(str) + " | R² " + tabela["r2"].astype(str)
tabela["observacao"] = tabela["numero"].map(OBS).fillna("")
tabela = tabela[["numero", "o_que_mudou", "resultado", "observacao", "algoritmo", "rmse", "mae", "r2", "tempo_treino_s", "publicavel"]]
tabela.to_csv("../modelos/tabela_experimentos.csv", index=False)
tabela

# %% [markdown]
# ## 6. Escolha do modelo publicado
# Entre os experimentos treinados com todas as colunas e todas as linhas, fica o de menor RMSE.

# %%
cand = tabela[tabela["publicavel"]].sort_values("rmse")
vencedor = int(cand.iloc[0]["numero"])
print("experimento publicado:", vencedor)
print(cand.iloc[0][["algoritmo", "o_que_mudou", "resultado", "tempo_treino_s"]])

# %%
pipeline_final = clone(pipelines[vencedor])
pipeline_final.fit(X_tr, y_tr)
joblib.dump(pipeline_final, "../modelos/modelo_publicado.joblib", compress=3)

info = {
    "tarefa": "regressao",
    "experimento": vencedor,
    "algoritmo": cand.iloc[0]["algoritmo"],
    "o_que_mudou": cand.iloc[0]["o_que_mudou"],
    "metricas": {k: float(cand.iloc[0][k]) for k in ["rmse", "mae", "r2"]},
    "alvo": "price (preço em US$)",
    "colunas_numericas": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in NUM},
    "colunas_categoricas": {c: sorted(X[c].dropna().unique().tolist()) for c in CAT},
}
# a ordem de qualidade de lapidação/cor/pureza fica da pior para a melhor, para o app mostrar na ordem certa
info["colunas_categoricas"]["cut"] = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
info["colunas_categoricas"]["color"] = ["J", "I", "H", "G", "F", "E", "D"]
info["colunas_categoricas"]["clarity"] = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
json.dump(info, open("../modelos/info_modelo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("modelo e info salvos em ../modelos/")

# %% [markdown]
# ## 7. Teste rápido do modelo salvo (mesmo caminho que o app usa)

# %%
modelo = joblib.load("../modelos/modelo_publicado.joblib")
exemplo = X_te.iloc[[0]]
print(exemplo.T)
print("previsão: US$", round(float(modelo.predict(exemplo)[0]), 2), "| real: US$", y_te.iloc[0])
