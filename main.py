# main.py
import yaml
import os
from src.logger import get_logger
from src.data_loader import load_and_clean_data
from src.visualizer import plot_categorical_counts, plot_correlation_heatmap, plot_numerical_distributions
from src.model import train_and_save_model
from src.evaluate import evaluate_model

# Load Config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Setup Main Logger
logger = get_logger("Main_Pipeline")

def main():
    logger.info(f"=== Starting {config['project_name']} ===")
    
    # Ensure output directories exist
    os.makedirs(config['paths']['outputs'], exist_ok=True)
    os.makedirs(config['paths']['models'], exist_ok=True)

    # Phase 1: Data Loading
    logger.info("Phase 1: Loading Data...")
    df = load_and_clean_data(config['paths']['data'])

    # Phase 2: EDA
    logger.info("Phase 2: Generating Visualizations...")
    plot_categorical_counts(df, output_path=os.path.join(config['paths']['outputs'], 'categorical_counts.png'))
    plot_correlation_heatmap(df, output_path=os.path.join(config['paths']['outputs'], 'correlation_heatmap.png'))
    plot_numerical_distributions(df, output_path=os.path.join(config['paths']['outputs'], 'numerical_distributions.png'))

    # Phase 3: Modeling
    logger.info("Phase 3: Training Model...")
    best_model, X_test, y_test, feature_names = train_and_save_model(df, config)

    # Phase 4: Evaluation
    logger.info("Phase 4: Evaluating Model...")
    evaluate_model(best_model, X_test, y_test, feature_names, config)

    logger.info("=== Pipeline Completed Successfully! ===")

if __name__ == "__main__":
    main()