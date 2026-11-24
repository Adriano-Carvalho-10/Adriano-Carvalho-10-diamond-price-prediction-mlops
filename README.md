# Previsão do preço de diamantes

3ª avaliação de MLOps. Tarefa de **regressão**: dadas as características de um diamante, prever o preço em dólares.

- **Base:** Diamonds (≈54 mil diamantes, 10 colunas). Alvo: `price`. Depois da limpeza (duplicadas e medidas inválidas) ficam 53.772 linhas.
- **Experimentos:** 7, na tabela `modelos/tabela_experimentos.csv` (e no notebook). 4 algoritmos (Regressão Linear, KNN, Random Forest, Gradient Boosting) e 3 experimentos que mudam os dados (sem a coluna `y`, metade das linhas e alvo em escala logarítmica). Todos avaliados no mesmo teste (20%).
- **Modelo publicado:** experimento 3, Random Forest com 100 árvores (RMSE US$ 538, MAE US$ 273, R² 0,981), salvo com joblib dentro de um Pipeline do scikit-learn.
- **App:** Streamlit. Recebe as características de um diamante, mostra o preço previsto e grava cada previsão em `registro_previsoes.csv` (data e hora, entradas, resultado e número do experimento).

## Organização

```
dados/       a base usada (diamonds.csv)
notebooks/   os experimentos (experimentos.ipynb e experimentos.py)
modelos/     modelo publicado (joblib), info_modelo.json e tabela de experimentos
app.py       o aplicativo Streamlit
requirements.txt
README.md
```

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para refazer os experimentos e regerar o modelo, abra `notebooks/experimentos.ipynb` e rode todas as células (precisa de `jupyter`). O notebook deve ser executado de dentro da pasta `notebooks/`. O Random Forest leva cerca de 15 segundos para treinar.

Fonte da base: https://github.com/mwaskom/seaborn-data (diamonds.csv)
