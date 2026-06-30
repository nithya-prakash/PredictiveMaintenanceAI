import numpy as np
import pandas as pd
import os
import argparse
from typing import Tuple

def generate_machine_data(machine_id: int, max_cycles: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic sensor data for a single machine over a number of cycles.
    Simulates degradation ending in failure.
    """
    np.random.seed(seed + machine_id)
    
    # Randomly determine the actual lifespan of this machine
    lifespan = np.random.randint(int(max_cycles * 0.6), max_cycles)
    cycles = np.arange(1, lifespan + 1)
    
    # Base operational parameters
    df = pd.DataFrame({
        'machine_id': machine_id,
        'cycle': cycles,
        'temperature': np.random.normal(70, 2, lifespan),
        'pressure': np.random.normal(100, 5, lifespan),
        'vibration': np.random.normal(0.5, 0.05, lifespan),
        'rpm': np.random.normal(1500, 50, lifespan),
        'voltage': np.random.normal(220, 2, lifespan),
        'current': np.random.normal(15, 0.5, lifespan),
        'oil_quality': np.linspace(100, 10 + np.random.normal(0, 5), lifespan) # linear degradation
    })
    
    # Inject non-linear degradation as the machine approaches failure
    degradation_factor = (cycles / lifespan) ** 3
    
    df['temperature'] += degradation_factor * np.random.normal(15, 2, lifespan)
    df['pressure'] += degradation_factor * np.random.normal(-20, 3, lifespan)
    df['vibration'] += degradation_factor * np.random.normal(1.5, 0.2, lifespan)
    df['rpm'] -= degradation_factor * np.random.normal(200, 20, lifespan)
    df['voltage'] -= degradation_factor * np.random.normal(10, 1, lifespan)
    df['current'] += degradation_factor * np.random.normal(5, 1, lifespan)
    
    # Add target columns
    df['RUL'] = lifespan - cycles
    
    # Binary classification target (failure within the next 30 cycles)
    failure_window = 30
    df['failure_imminent'] = (df['RUL'] <= failure_window).astype(int)
    
    # Multi-class failure mode (0: healthy, 1: bearing failure, 2: pump failure, 3: electrical failure)
    failure_type = np.zeros(lifespan)
    if lifespan > 0:
        # Pick a random failure type for this machine
        f_type = np.random.choice([1, 2, 3])
        failure_type[df['RUL'] <= failure_window] = f_type
    df['failure_type'] = failure_type.astype(int)
    
    return df

def generate_dataset(num_machines: int = 100, max_cycles: int = 250, output_dir: str = 'datasets') -> None:
    """
    Generates a full dataset containing multiple machines and saves it to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic data for {num_machines} machines...")
    dataframes = []
    
    for i in range(1, num_machines + 1):
        df = generate_machine_data(machine_id=i, max_cycles=max_cycles, seed=42)
        dataframes.append(df)
        
    full_dataset = pd.concat(dataframes, ignore_index=True)
    
    # Split into train and test
    train_machines = int(num_machines * 0.8)
    
    train_df = full_dataset[full_dataset['machine_id'] <= train_machines]
    test_df = full_dataset[full_dataset['machine_id'] > train_machines]
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Generated {len(train_df)} rows for training (saved to {train_path})")
    print(f"Generated {len(test_df)} rows for testing (saved to {test_path})")
    print("Dataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic predictive maintenance dataset.")
    parser.add_argument('--machines', type=int, default=150, help='Number of machines to simulate')
    parser.add_argument('--max-cycles', type=int, default=300, help='Maximum operational cycles per machine')
    parser.add_argument('--output-dir', type=str, default='datasets', help='Directory to save the CSV files')
    
    args = parser.parse_args()
    
    generate_dataset(num_machines=args.machines, max_cycles=args.max_cycles, output_dir=args.output_dir)
