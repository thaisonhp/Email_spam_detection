import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessing

class ModelPredictor:
    def __init__(self):
        # Tải mô hình và vectorizer đã huấn luyện
        self.model = joblib.load('models/naive_bayes_model.joblib')
        self.vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        self.processor = DataPreprocessing()

    def predict(self, text):
        # Tiền xử lý và vectorize dữ liệu đầu vào
        cleaned_text = self.processor.clean_text(pd.DataFrame({"v2": [text]}), text_column="v2")
        vectorized_input = self.vectorizer.transform(cleaned_text["v2"])

        # Dự đoán
        prediction = self.model.predict(vectorized_input)
        return prediction[0]
