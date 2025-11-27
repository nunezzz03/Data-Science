"""
Utility functions for the data preparation pipeline
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import lab3_config as config


def evaluate_models(X_train, y_train, X_test, y_test, approach_name=""):
    """
    Train and evaluate KNN and Naive Bayes models using F1 score.
    Returns F1 scores for both models and their average.
    """
    results = {}
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['NB'] = f1_score(y_test, y_pred_nb, pos_label=True, average='binary', zero_division=0)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = f1_score(y_test, y_pred_knn, pos_label=True, average='binary', zero_division=0)
    
    # Average F1
    results['AVG'] = (results['NB'] + results['KNN']) / 2
    
    print(f"         {approach_name}: NB F1={results['NB']:.4f}, KNN F1={results['KNN']:.4f}, AVG F1={results['AVG']:.4f}")
    
    return results


def save_dataset(X_train, y_train, X_test, y_test, filepath, metadata=None):
    """
    Save train and test datasets to a single CSV file with a split indicator.
    Also saves metadata about the dataset.
    """
    # Add split indicator
    train_df = X_train.copy()
    train_df['_split'] = 'train'
    train_df[config.TARGET] = y_train
    
    test_df = X_test.copy()
    test_df['_split'] = 'test'
    test_df[config.TARGET] = y_test
    
    # Combine and save
    combined = pd.concat([train_df, test_df], axis=0)
    combined.to_csv(filepath, index=False)
    
    print(f"         Saved: {filepath}")
    print(f"         Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Save metadata if provided
    if metadata:
        meta_file = filepath.replace('.csv', '_metadata.txt')
        with open(meta_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")


def load_dataset(filepath):
    """
    Load train and test datasets from a CSV file with split indicator.
    """
    df = pd.read_csv(filepath)
    
    # Split back into train and test
    train_df = df[df['_split'] == 'train'].drop(columns=['_split'])
    test_df = df[df['_split'] == 'test'].drop(columns=['_split'])
    
    # Separate X and y
    y_train = train_df[config.TARGET]
    X_train = train_df.drop(columns=[config.TARGET])
    
    y_test = test_df[config.TARGET]
    X_test = test_df.drop(columns=[config.TARGET])
    
    return X_train, y_train, X_test, y_test


def plot_comparison(results_dict, step_name, save_path):
    """
    Create a bar chart comparing F1 scores of different approaches.
    """
    approaches = list(results_dict.keys())
    nb_scores = [results_dict[a]['NB'] for a in approaches]
    knn_scores = [results_dict[a]['KNN'] for a in approaches]
    avg_scores = [results_dict[a]['AVG'] for a in approaches]
    
    x = range(len(approaches))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], nb_scores, width, label='Naive Bayes', alpha=0.8)
    ax.bar(x, knn_scores, width, label='KNN', alpha=0.8)
    ax.bar([i + width for i in x], avg_scores, width, label='Average', alpha=0.8)
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'{step_name} - Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"         Chart saved: {save_path}")


def random_oversampling(X, y):
    """Random oversampling of minority class."""
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    df_minority_over = df_minority.sample(len(df_majority), replace=True, random_state=config.RANDOM_STATE)
    
    df_balanced = pd.concat([df_majority, df_minority_over], axis=0)
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    return X_balanced, y_balanced


def random_undersampling(X, y):
    """Random undersampling of majority class."""
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    df_majority_under = df_majority.sample(len(df_minority), replace=False, random_state=config.RANDOM_STATE)
    
    df_balanced = pd.concat([df_majority_under, df_minority], axis=0)
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    return X_balanced, y_balanced
