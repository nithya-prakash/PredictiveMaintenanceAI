import os
import json
import time
import joblib
import pandas as pd
import numpy as np

def evaluate_latency():
    print("Evaluating Inference Latency...")
    rf_classifier = joblib.load("models/rf_classifier.pkl")
    rf_rul = joblib.load("models/rf_rul.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    df_test = pd.read_csv("datasets/test_data.csv").head(150) 
    # Benchmarking API request pipeline internally without network overhead
    
    latencies = {"model_only_classification": [], "full_pipeline_classification": []}
    
    # Warmup
    for i in range(10):
        preprocessor.add_rolling_features(df_test.iloc[0:i+1])
        
    for i in range(15, 115):
        start_full = time.perf_counter()
        
        # Simulate an incoming stream of recent cycles for one machine
        df_stream = df_test.iloc[0:i].copy()
        df_feat = preprocessor.add_rolling_features(df_stream)
        df_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_feat[preprocessor.feature_cols])
        
        sample = df_feat[preprocessor.feature_cols].values[-1].reshape(1, -1)
        
        start_model = time.perf_counter()
        _ = rf_classifier.predict(sample)
        latencies["model_only_classification"].append((time.perf_counter() - start_model) * 1000)
        
        latencies["full_pipeline_classification"].append((time.perf_counter() - start_full) * 1000)
        
    metrics = {}
    for key, times in latencies.items():
        metrics[key] = {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99))
        }
        
    metrics["benchmark_context"] = "Local execution, 100 sequential requests, 10 warmup calls. Simulates preprocessing + scaling + inference overhead."
        
    with open("evaluation/results/latency.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Latency evaluation complete.")

if __name__ == "__main__":
    evaluate_latency()
