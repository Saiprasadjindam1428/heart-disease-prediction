import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_counts(df, output_path="categorical_counts.png"):
    """Generates and saves a count plot for all categorical features."""
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        if col in df.columns:
            sns.countplot(data=df, x=col, hue=col, ax=axes[i], palette='viridis', legend=False)
            axes[i].set_title(f'Count of {col}')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Categorical count plot saved to {output_path}")

def plot_correlation_heatmap(df, output_path="correlation_heatmap.png"):
    """Generates and saves a correlation heatmap for numerical features."""
    plt.figure(figsize=(12, 10))
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Generate the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def plot_numerical_distributions(df, output_path="numerical_distributions.png"):
    """Generates and saves histograms for numerical features, grouped by target."""
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    cols_to_plot = [col for col in num_cols if col in df.columns]
    
    if not cols_to_plot:
        print("No standard numerical columns found to plot.")
        return

    n_cols = len(cols_to_plot)
    rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        sns.histplot(data=df, x=col, hue='target', kde=True, ax=axes[i], palette='Set1', alpha=0.6)
        axes[i].set_title(f'Distribution of {col} by Target')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Numerical distribution plots saved to {output_path}")