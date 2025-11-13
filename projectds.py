# projectds.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_traffic_accidents(file_path='/Users/jnunezzz03/Desktop/data science/projectds/datasets/traffic_accidents.csv', target_column='crash_type'):
    """
    Load Traffic Accidents dataset and apply the required cleaning for Lab 1
    """
    print("=== LOADING AND CLEANING TRAFFIC ACCIDENTS DATASET ===\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at {file_path}")
        return None, None, None
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check target variable
    if target_column in df.columns:
        print(f"🎯 Target variable '{target_column}' found")
        print(f"📈 Target distribution:\n{df[target_column].value_counts()}")
    else:
        print(f"❌ ERROR: Target column '{target_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None
    
    # 1. Drop completely empty variables
    df_clean = df.dropna(axis=1, how='all')
    print(f"✅ After dropping empty variables: {df_clean.shape}")
    
    # 2. Encode target variable to numeric before dropping non-numeric columns
    le = LabelEncoder()
    df_clean[target_column] = le.fit_transform(df_clean[target_column])
    print(f"🔢 Target variable encoded to numeric")
    print(f"   Target classes mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # 3. Drop all non-numeric variables (except the target which we just encoded)
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Make sure target is included
    if target_column not in numeric_columns:
        numeric_columns.append(target_column)
    
    numeric_df = df_clean[numeric_columns]
    print(f"✅ After dropping non-numeric variables: {numeric_df.shape}")
    
    # Show which columns were kept
    print(f"🔢 Numeric columns kept: {list(numeric_df.columns)}")
    
    # 4. Drop records with any missing values
    final_df = numeric_df.dropna()
    print(f"✅ After dropping records with missing values: {final_df.shape}")
    
    # Check if we still have the target column
    if target_column not in final_df.columns:
        print(f"❌ ERROR: Target column '{target_column}' was lost during cleaning!")
        return None, None, None
    
    # Separate features and target
    X = final_df.drop(columns=[target_column])
    y = final_df[target_column]
    
    print(f"\n🎯 Final dataset:")
    print(f"   Features: {X.shape[1]} variables")
    print(f"   Samples: {X.shape[0]} records")
    print(f"   Target classes: {len(y.unique())}")
    print(f"   Target distribution:\n{y.value_counts()}")
    print(f"   Features used: {list(X.columns)}")
    
    return X, y, final_df

def train_baseline_models(X, y, dataset_name="Traffic Accidents"):
    """
    Train all baseline models and evaluate their performance
    """
    print(f"\n=== TRAINING BASELINE MODELS FOR {dataset_name.upper()} ===\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Training target distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"   Test target distribution:\n{pd.Series(y_test).value_counts()}")
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'MLP': MLPClassifier(max_iter=1000, random_state=42, hidden_layer_sizes=(100,))
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Use scaled data for certain models
        if name in ['KNN', 'Logistic Regression', 'MLP']:
            X_tr, X_te = X_train_scaled, X_test_scaled
            print("   Using scaled features")
        else:
            X_tr, X_te = X_train, X_test
            print("   Using original features")
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = model.predict(X_te)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall: {recall:.4f}")
        
        # Show best parameters for some models
        if name == 'Decision Tree' and hasattr(model, 'get_depth'):
            print(f"   🌳 Tree depth: {model.get_depth()}")
        elif name == 'KNN':
            print(f"   📍 Number of neighbors: {model.n_neighbors}")
    
    return results, X_train, X_test, y_train, y_test

def create_performance_chart(results, dataset_name):
    """
    Create performance comparison chart
    """
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, precisions, width, label='Precision', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, recalls, width, label='Recall', color='lightcoral', edgecolor='black')
    
    # Customize chart
    ax.set_xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Scores', fontsize=12, fontweight='bold')
    ax.set_title(f'Baseline Models Performance - {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    return fig

def create_metrics_table(results, dataset_name):
    """
    Create a detailed metrics table
    """
    metrics_data = []
    for model_name, metrics in results.items():
        metrics_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}", 
            'Recall': f"{metrics['recall']:.4f}"
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_metrics.values,
                    colLabels=df_metrics.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title(f'Performance Metrics - {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    
    return fig, df_metrics

def create_hyperparameter_studies(X, y, results):
    """
    Create hyperparameter study charts for specific models
    """
    print("\n📊 Generating hyperparameter studies...")
    
    # KNN Hyperparameter Study
    knn_accuracies = []
    k_values = range(1, 21)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        knn_accuracies.append(accuracy)
    
    # Plot KNN study
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # KNN plot
    ax1.plot(k_values, knn_accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('KNN: Hyperparameter Study (k vs Accuracy)')
    ax1.grid(True, alpha=0.3)
    
    # Mark the best k
    best_k = k_values[np.argmax(knn_accuracies)]
    best_accuracy = max(knn_accuracies)
    ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k} (Acc: {best_accuracy:.3f})')
    ax1.legend()
    
    # Decision Tree depths comparison
    tree_depths = [5, 10, 15, 20, None]
    tree_accuracies = []
    
    for depth in tree_depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        tree_accuracies.append(accuracy)
    
    # Plot Decision Tree study
    ax2.bar([str(d) if d else 'None' for d in tree_depths], tree_accuracies, color='orange', alpha=0.7)
    ax2.set_xlabel('Max Tree Depth')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Decision Tree: Hyperparameter Study (Depth vs Accuracy)')
    ax2.grid(True, alpha=0.3)
    
    # Mark the best depth
    best_depth_idx = np.argmax(tree_accuracies)
    best_depth = tree_depths[best_depth_idx]
    best_tree_accuracy = tree_accuracies[best_depth_idx]
    ax2.axhline(y=best_tree_accuracy, color='red', linestyle='--', alpha=0.7, 
                label=f'Best depth={best_depth} (Acc: {best_tree_accuracy:.3f})')
    ax2.legend()
    
    plt.tight_layout()
    return fig1

def main():
    """
    Main function to run the baseline analysis for Traffic Accidents
    """
    print("🚗 TRAFFIC ACCIDENTS - BASELINE MODELS ANALYSIS")
    print("=" * 50)
    
    # 1. Load and clean data
    X, y, clean_df = load_and_clean_traffic_accidents()
    
    if X is None:
        print("❌ Failed to load and clean data. Exiting.")
        return
    
    # 2. Train baseline models
    results, X_train, X_test, y_train, y_test = train_baseline_models(X, y)
    
    # 3. Create performance chart
    print("\n📈 Generating performance charts...")
    performance_fig = create_performance_chart(results, "Traffic Accidents")
    
    # 4. Create metrics table
    metrics_fig, metrics_df = create_metrics_table(results, "Traffic Accidents")
    
    # 5. Create hyperparameter studies
    hyperparam_fig = create_hyperparameter_studies(X, y, results)
    
    # 6. Save results
    print("\n💾 Saving results...")
    
    # Define output directory
    output_dir = '/Users/jnunezzz03/Desktop/data science/projectds/'
    
    # Save charts
    performance_fig.savefig(output_dir + 'traffic_accidents_performance.png', dpi=300, bbox_inches='tight')
    metrics_fig.savefig(output_dir + 'traffic_accidents_metrics_table.png', dpi=300, bbox_inches='tight')
    hyperparam_fig.savefig(output_dir + 'traffic_accidents_hyperparameter_study.png', dpi=300, bbox_inches='tight')
    
    # Save metrics to CSV
    metrics_df.to_csv(output_dir + 'traffic_accidents_metrics.csv', index=False)
    
    # Save best model parameters
    best_model_info = []
    for model_name, result in results.items():
        model = result['model']
        best_model_info.append({
            'Model': model_name,
            'Best_Parameters': str(model.get_params()),
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall']
        })
    
    best_model_df = pd.DataFrame(best_model_info)
    best_model_df.to_csv(output_dir + 'traffic_accidents_best_models.csv', index=False)
    
    print("✅ Analysis completed!")
    print("📁 Files generated:")
    print("   - traffic_accidents_performance.png")
    print("   - traffic_accidents_metrics_table.png") 
    print("   - traffic_accidents_hyperparameter_study.png")
    print("   - traffic_accidents_metrics.csv")
    print("   - traffic_accidents_best_models.csv")
    
    # Show summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_model = min(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 Best model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    print(f"📉 Worst model: {worst_model[0]} (Accuracy: {worst_model[1]['accuracy']:.4f})")
    
    # Display charts
    plt.show()

# Run the analysis
if __name__ == "__main__":
    main()