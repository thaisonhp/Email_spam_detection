from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.model = joblib.load('models/naive_bayes_model.joblib')

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
