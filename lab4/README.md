# Lab 4 - Modelling and Overfitting

## Objetivo

Treinar modelos de classificação com dados preparados e avaliar o seu desempenho.

## Modelos Implementados

1. **Naive Bayes** - GaussianNB, BernoulliNB
2. **Logistic Regression** - L1/L2 regularization
3. **K-Nearest Neighbors (KNN)** - Diferentes valores de K e métricas de distância
4. **Decision Trees** - Entropy/Gini, diferentes profundidades
5. **Multi-Layer Perceptron (MLP)** - Diferentes arquiteturas e learning rates
6. **Random Forest** - Número de estimadores e profundidade
7. **Gradient Boosting / XGBoost** - Learning rate e número de estimadores

## Estrutura

```
lab4/
├── config.py           # Configurações centralizadas
├── prepare_data.py     # Preparação dos dados (com fix de data leakage)
├── run_all.py          # Script principal para correr tudo
├── run_all.sh          # Script shell alternativo
├── summary.py          # Comparação e sumário dos modelos
├── models/
│   ├── naive_bayes.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── mlp.py
│   ├── random_forest.py
│   └── gradient_boosting.py
├── data/               # Dados processados (gerados)
├── images/             # Gráficos gerados
└── results/            # CSVs com resultados
```

## Como Executar

### Opção 1: Script Python

```bash
cd lab4
python3 run_all.py
```

### Opção 2: Script Shell

```bash
cd lab4
./run_all.sh
```

### Opção 3: Executar modelos individualmente

```bash
cd lab4
python3 prepare_data.py          # Primeiro, preparar dados
python3 models/naive_bayes.py    # Correr modelo específico
python3 summary.py               # Gerar sumário
```

## Outputs Gerados

### Imagens (por modelo e dataset)

- **Hyperparameters Study** - Gráficos mostrando performance vs hiperparâmetros
- **Overfitting Study** - Comparação train vs test
- **Feature Importance** - Para modelos baseados em árvores e regressão logística
- **Model Comparison** - Comparação entre todos os modelos

### Resultados (CSVs)

- `{dataset}_summary.csv` - Sumário por dataset
- `overall_summary.csv` - Sumário geral
- `{model}_results.csv` - Resultados detalhados por modelo

## Data Leakage Fix

O script `prepare_data.py` remove automaticamente as colunas que causam data leakage:

- Informação pós-chegada (ArrTime, ArrDelay, etc.)
- Informação pós-partida (DepTime, DepDelay, etc.)
- Colunas redundantes ou sem informação útil

## Métricas Avaliadas

- **Accuracy** - Percentagem de previsões corretas
- **Precision** - Previsões positivas corretas / Total previsões positivas
- **Recall** - Previsões positivas corretas / Total positivos reais
- **F1 Score** - Média harmónica de Precision e Recall

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib
pip install xgboost  # Opcional, para XGBoost
```
