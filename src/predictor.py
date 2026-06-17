# load the best model version from each model type 
# construct class prdictor to hide the thresholding detail and reuse the final model from a custom class 

import joblib 
import numpy as np 

class ChurnModel:

    def __init__(self, model, threshold):
        self.model = model 
        self.threshold = threshold 

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return np.where(probs >= self.threshold, 1, 0)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    


