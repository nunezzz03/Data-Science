import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import sys
import os

# =======================================================================================
# 🔧 CORREÇÃO DE PATHS (Para resolver o erro "No module named 'config'")
# =======================================================================================
# Obter o caminho da raiz do projeto
# __file__ = .../classification/models/prepare_flights.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 1. Adicionar a raiz ao path (para encontrar 'utils')
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Adicionar a pasta 'utils' ao path (CRÍTICO: para o dslabs encontrar o config.py)
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Agora os imports já funcionam
from utils.dslabs_functions import (
    mvi_by_filling, 
    dummify, 
    select_low_variance_variables, 
    select_redundant_variables,
    determine_outlier_thresholds_for_var,
    get_variable_types
)

# =======================================================================================
# 🔄 FUNÇÃO DE BALANCEAMENTO MANUAL (Substitui o SMOTE para não precisares de instalar nada)
# =======================================================================================
def random_oversampling(X, y):
    """
    Realiza balanceamento duplicando exemplos da classe minoritária (Pandas puro).
    Evita o erro do 'imblearn'.
    """
    print(f"      ⚖️ A balancear classes (Random Oversampling)...")
    # Juntar X e y
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    # Contar as classes
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    # Separar os dados
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # Duplicar a classe minoritária até ter o mesmo tamanho da maioritária
    df_minority_over = df_minority.sample(len(df_majority), replace=True, random_state=42)
    
    # Juntar tudo
    df_balanced = pd.concat([df_majority, df_minority_over], axis=0)
    
    # Separar X e y novamente
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    print(f"         -> Classes balanceadas: {y_balanced.value_counts().to_dict()}")
    return X_balanced, y_balanced

# =======================================================================================
# 🚀 FUNÇÃO PRINCIPAL
# =======================================================================================
def prepare_flights_dataset():
    print("\n✈️ A INICIAR PREPARAÇÃO DO DATASET FLIGHTS (Estilo Lab)...")
    
    # --- 1. CARREGAMENTO E LEAKAGE ---
    filename = "data/raw/Combined_Flights_2022.csv"
    filepath = os.path.join(project_root, filename) # Caminho absoluto para não falhar
    
    if not os.path.exists(filepath):
        print(f"❌ Erro: Ficheiro não encontrado em {filepath}")
        return

    df = pd.read_csv(filepath, na_values="", parse_dates=True)
    
    # Amostragem (10% para rapidez)
    df = df.sample(frac=0.1, random_state=42)
    print(f"   1. Dados carregados e amostrados: {df.shape}")

    # Remover Leakage e IDs
    leakage_cols = [
        "ArrTime", "ArrDelayMinutes", "ArrDelay", "ActualElapsedTime",
        "WheelsOn", "TaxiIn", "ArrivalDelayGroups", "ArrTimeBlk",
        "FlightDate", "Tail_Number"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    target = "ArrDel15"
    
    # --- 2. MISSING VALUES (mvi_by_filling) ---
    # Usa a função do dslabs tal como no exemplo
    df = mvi_by_filling(df, strategy="frequent")
    print(f"   2. Missing Values tratados. Dimensão: {df.shape}")

    # --- 3. OUTLIERS (Abordagem da Professora) ---
    numeric_vars = get_variable_types(df)["numeric"]
    if target in numeric_vars: numeric_vars.remove(target)
    
    summary5 = df[numeric_vars].describe()
    initial_rows = df.shape[0]
    
    print("   3. A remover outliers...")
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var])
        # Identificar outliers
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        # Dropar outliers
        df.drop(outliers.index, axis=0, inplace=True)
        
    print(f"      -> Removidos {initial_rows - df.shape[0]} registos. Dimensão atual: {df.shape}")

    # --- 4. FEATURE SELECTION ---
    # Variância
    vars_to_drop = select_low_variance_variables(df, max_threshold=0.1, target=target)
    df = df.drop(columns=vars_to_drop)
    print(f"   4. Feature Selection completa (Low Variance). Variáveis: {df.shape[1]}")

    # --- 5. SCALING (StandardScaler) ---
    # Aplicar Z-Score normalization mantendo a estrutura do DataFrame
    # Recalcular variáveis numéricas porque algumas podem ter sido removidas no passo anterior
    numeric_vars = get_variable_types(df)["numeric"]
    if target in numeric_vars: numeric_vars.remove(target)
    
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df[numeric_vars])
    df_scaled = df.copy()
    df_scaled[numeric_vars] = scaler.transform(df[numeric_vars])
    print("   5. Scaling aplicado (StandardScaler).")

    # ============================================================
    # CRIAÇÃO DOS DATASETS FINAIS (Ordinal vs One-Hot)
    # ============================================================
    
    y = df_scaled[target]
    X = df_scaled.drop(columns=[target])
    symbolic_vars = get_variable_types(X)["symbolic"]

    # --- ABORDAGEM A: ONE-HOT ENCODING (dummify) ---
    print("\n   🅰️  A criar versão One-Hot Encoding...")
    X_onehot = dummify(X, symbolic_vars)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, train_size=0.7, stratify=y, random_state=42)
    
    # Balanceamento (Random Oversampling)
    X_train_bal, y_train_bal = random_oversampling(X_train, y_train)
    
    save_split("flights_onehot", X_train_bal, y_train_bal, X_test, y_test)

    # --- ABORDAGEM B: ORDINAL ENCODING ---
    print("\n   🅱️  A criar versão Ordinal Encoding...")
    X_ordinal = X.copy()
    enc = OrdinalEncoder()
    # Converter para string para garantir que o OrdinalEncoder não falha
    X_ordinal[symbolic_vars] = X_ordinal[symbolic_vars].astype(str)
    X_ordinal[symbolic_vars] = enc.fit_transform(X_ordinal[symbolic_vars])
    
    # Split
    X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(X_ordinal, y, train_size=0.7, stratify=y, random_state=42)
    
    # Balanceamento (Random Oversampling)
    X_train_ord_bal, y_train_ord_bal = random_oversampling(X_train_ord, y_train_ord)
    
    save_split("flights_ordinal", X_train_ord_bal, y_train_ord_bal, X_test_ord, y_test_ord)

    print("\n✅ FIM! Ficheiros guardados em data/processed/")

def save_split(prefix, X_train, y_train, X_test, y_test):
    output_dir = os.path.join(project_root, "data/processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    train.to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
    test.to_csv(f"{output_dir}/{prefix}_test.csv", index=False)
    print(f"      💾 Guardado: {prefix} (Train: {train.shape}, Test: {test.shape})")

if __name__ == "__main__":
    prepare_flights_dataset()