import shap
import pandas as pd
import numpy as np
from backend.services.predictor import predictor_service

class XAIService:
    def __init__(self):
        # We use TreeExplainer because we're using RandomForest models
        self.rul_explainer = None
        self.classifier_explainer = None
        
        # We need a small background dataset to initialize some explainers,
        # but TreeExplainer works fine without it (marginalizing over the tree).
        
    def _init_explainers(self):
        if predictor_service.model_rul and not self.rul_explainer:
            self.rul_explainer = shap.TreeExplainer(predictor_service.model_rul)
            
        if predictor_service.model_classifier and not self.classifier_explainer:
            self.classifier_explainer = shap.TreeExplainer(predictor_service.model_classifier)
            
    def explain_failure(self, data) -> dict:
        self._init_explainers()
        if not self.classifier_explainer:
            return {"error": "Model not loaded"}
            
        features = predictor_service._prepare_data([data])
        
        # Calculate SHAP values
        shap_values = self.classifier_explainer.shap_values(features)
        
        # Binary classification usually returns shape (1, num_features) or (1, num_features, 2)
        # For RandomForest, shape is (1, num_features, 2) where [:,:,1] is the positive class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
                
        feature_names = predictor_service.preprocessor.feature_cols
        
        importance_dict = {feat: float(val) for feat, val in zip(feature_names, sv)}
        
        # Sort by absolute impact
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        top_contributor = list(sorted_importance.keys())[0] if sorted_importance else "Unknown"
        
        return {
            "feature_importance": sorted_importance,
            "top_contributor": top_contributor
        }

xai_service = XAIService()
