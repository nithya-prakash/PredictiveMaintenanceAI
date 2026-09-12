import joblib
import pandas as pd
import os

class PredictorService:
    def __init__(self):
        self.preprocessor = None
        self.model_rul = None
        self.model_classifier = None
        self.classifier_algorithm = None
        self.classifier_background = None
        self.model_anomaly = None
        self._load_models()

    def _load_models(self):
        base_dir = "models"
        try:
            self.preprocessor = joblib.load(os.path.join(base_dir, "preprocessor.pkl"))
            self.model_rul = joblib.load(os.path.join(base_dir, "rf_rul.pkl"))

            classifier_artifact = joblib.load(os.path.join(base_dir, "classifier.pkl"))
            # Whichever of the RandomForest/LogisticRegression candidates won on
            # PR-AUC during training is what's saved here — see ml/training/train_classifier.py.
            self.model_classifier = classifier_artifact["model"]
            self.classifier_algorithm = classifier_artifact["algorithm"]
            self.classifier_background = classifier_artifact.get("background_sample")

            self.model_anomaly = joblib.load(os.path.join(base_dir, "isolation_forest.pkl"))
            print(f"Successfully loaded all models. Classifier: {self.classifier_algorithm}")
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def _prepare_data(self, data_list):
        df = pd.DataFrame([d.dict() for d in data_list])
        df_feat = self.preprocessor.add_rolling_features(df)
        
        # In a real streaming scenario, rolling features need historical context.
        # For simplicity here, we assume the batch has enough context or we just use raw features.
        
        # Transform
        df_feat[self.preprocessor.feature_cols] = self.preprocessor.scaler.transform(df_feat[self.preprocessor.feature_cols])
        return df_feat[self.preprocessor.feature_cols]

    def predict_rul(self, data):
        features = self._prepare_data([data])
        pred = self.model_rul.predict(features)[0]
        return float(pred)
        
    def predict_failure(self, data):
        features = self._prepare_data([data])
        prob = self.model_classifier.predict_proba(features)[0][1]
        return float(prob)
        
    def detect_anomaly(self, data):
        features = self._prepare_data([data])
        pred = self.model_anomaly.predict(features)[0]
        return pred == -1 # IsolationForest returns -1 for anomalies

    def get_anomaly_score(self, data):
        # Continuous IsolationForest score: lower (more negative) = more anomalous.
        features = self._prepare_data([data])
        return float(self.model_anomaly.decision_function(features)[0])


predictor_service = PredictorService()
