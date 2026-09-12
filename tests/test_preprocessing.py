import pandas as pd
import numpy as np
from ml.features.preprocessing import PredictiveMaintenancePreprocessor

def test_rolling_features_no_leakage():
    """
    Ensures that rolling features do not leak data across different machines.
    """
    # Create dummy data for two machines
    df = pd.DataFrame({
        'machine_id': [1, 1, 1, 2, 2],
        'cycle': [1, 2, 3, 1, 2],
        'temperature': [10, 20, 30, 100, 200],
        'pressure': [1, 1, 1, 1, 1],
        'vibration': [1, 1, 1, 1, 1],
        'rpm': [1, 1, 1, 1, 1],
        'voltage': [1, 1, 1, 1, 1],
        'current': [1, 1, 1, 1, 1],
        'oil_quality': [1, 1, 1, 1, 1],
        'RUL': [5, 4, 3, 5, 4],
        'failure_imminent': [0, 0, 0, 0, 0],
        'failure_type': [0, 0, 0, 0, 0]
    })
    
    preprocessor = PredictiveMaintenancePreprocessor()
    df_feat = preprocessor.add_rolling_features(df, window_sizes=[2])
    
    # Machine 2, cycle 1 should only have rolling mean of 100 (not mixed with Machine 1)
    machine_2_cycle_1 = df_feat[(df_feat['machine_id'] == 2) & (df_feat['cycle'] == 1)]
    assert machine_2_cycle_1['temperature_roll_mean_2'].values[0] == 100.0, "Leakage detected across machines!"
    
    # Machine 2, cycle 2 should have rolling mean of 150
    machine_2_cycle_2 = df_feat[(df_feat['machine_id'] == 2) & (df_feat['cycle'] == 2)]
    assert machine_2_cycle_2['temperature_roll_mean_2'].values[0] == 150.0, "Rolling feature calculated incorrectly!"
