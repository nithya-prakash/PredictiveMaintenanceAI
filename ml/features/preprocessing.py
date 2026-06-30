import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict
import joblib
import os

class PredictiveMaintenancePreprocessor:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        self.sensor_cols = [
            'temperature', 'pressure', 'vibration', 'rpm', 
            'voltage', 'current', 'oil_quality'
        ]
        
    def add_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 15]) -> pd.DataFrame:
        """
        Adds rolling mean and standard deviation features.
        """
        df_out = df.copy()
        
        for w in window_sizes:
            for col in self.sensor_cols:
                # Group by machine_id to prevent leaking across machines
                roll = df_out.groupby('machine_id')[col].rolling(window=w, min_periods=1)
                df_out[f'{col}_roll_mean_{w}'] = roll.mean().reset_index(level=0, drop=True)
                df_out[f'{col}_roll_std_{w}'] = roll.std().reset_index(level=0, drop=True).fillna(0)
                
        return df_out
        
    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Fits the scaler and transforms the dataset. Adds engineered features.
        """
        df_feat = self.add_rolling_features(df)
        
        # Get all feature columns (sensors + rolling)
        self.feature_cols = [c for c in df_feat.columns if c not in ['machine_id', 'cycle', 'RUL', 'failure_imminent', 'failure_type']]
        
        if is_training:
            df_feat[self.feature_cols] = self.scaler.fit_transform(df_feat[self.feature_cols])
        else:
            df_feat[self.feature_cols] = self.scaler.transform(df_feat[self.feature_cols])
            
        return df_feat
        
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts the dataframe into 3D sequences (samples, sequence_length, features) 
        for models like LSTM/Transformers.
        Returns: X (sequences), y_rul (Remaining Useful Life), y_class (failure_imminent)
        """
        sequence_cols = self.feature_cols
        
        seqs = []
        labels_rul = []
        labels_class = []
        
        for machine_id in df['machine_id'].unique():
            machine_data = df[df['machine_id'] == machine_id]
            
            data_matrix = machine_data[sequence_cols].values
            rul_matrix = machine_data['RUL'].values
            class_matrix = machine_data['failure_imminent'].values
            
            # We can only create sequences if the machine has enough cycles
            for i in range(len(data_matrix) - self.sequence_length + 1):
                seqs.append(data_matrix[i:i+self.sequence_length])
                
                # The label is the target at the LAST time step of the sequence
                labels_rul.append(rul_matrix[i+self.sequence_length-1])
                labels_class.append(class_matrix[i+self.sequence_length-1])
                
        return np.array(seqs), np.array(labels_rul), np.array(labels_class)
        
    def save(self, filepath: str):
        """Saves the fitted preprocessor (scaler)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> 'PredictiveMaintenancePreprocessor':
        """Loads a fitted preprocessor"""
        return joblib.load(filepath)

if __name__ == "__main__":
    # Test the preprocessor
    train_path = "datasets/train_data.csv"
    if os.path.exists(train_path):
        print("Testing preprocessing pipeline...")
        df_train = pd.read_csv(train_path)
        preprocessor = PredictiveMaintenancePreprocessor(sequence_length=15)
        
        df_train_feat = preprocessor.fit_transform(df_train, is_training=True)
        print(f"Engineered features: {len(preprocessor.feature_cols)}")
        
        X_seq, y_rul, y_class = preprocessor.create_sequences(df_train_feat)
        print(f"Created sequences: {X_seq.shape}")
        
        preprocessor.save("models/preprocessor.pkl")
        print("Saved preprocessor to models/preprocessor.pkl")
    else:
        print(f"{train_path} not found. Run generate_dataset.py first.")
