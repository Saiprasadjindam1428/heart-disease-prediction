import pandas as pd
from pathlib import Path

def load_and_clean_data(file_path):
    """Loads dataset from excel and performs initial cleaning."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please check the path.")
        
    df = pd.read_excel(path)
    
    # Drop duplicates as per the original notebook
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    return df