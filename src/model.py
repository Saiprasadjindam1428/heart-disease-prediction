# src/model.py
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.logger import get_logger

logger = get_logger(__name__)

def train_and_save_model(df, config):
    """Trains the model, tunes parameters, and saves the final model to disk."""
    target_col = config['model_params']['target_column']
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model_params']['test_size'], 
        random_state=config['model_params']['random_state']
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=5000, random_state=config['model_params']['random_state']))
    ])

    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__penalty': ['l2'],
        'logreg__solver': ['lbfgs']
    }

    logger.info("Starting Grid Search for Hyperparameter Tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best Model Found. Parameters: {grid_search.best_params_}")
    
    # Save the model
    model_path = os.path.join(config['paths']['models'], 'best_logistic_model.joblib')
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved successfully to {model_path}")
    
    return best_model, X_test, y_test, X.columns