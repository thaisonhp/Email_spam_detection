# main.py
import pandas as pd

from src.train import ModelTrainer
from src.predict import ModelPredictor
from src.evaluate import ModelEvaluator

def main():
    # Bước 1: Huấn luyện mô hình
    raw_data = pd.read_csv(r"data /raw/spam.csv", encoding='ISO-8859-1')
    trainer = ModelTrainer()
    trainer.train(raw_data)

    # Bước 2: Dự đoán với một văn bản
    predictor = ModelPredictor()
    prediction = predictor.predict("Hey thaisonhp! A third-party OAuth application was recently authorized to access your account.")
    print("Prediction:", prediction)

    # Bước 3: Đánh giá mô hình
    evaluator = ModelEvaluator()
    # Ví dụ với dữ liệu test, giả sử bạn đã có X_test, y_test
    # evaluator.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
