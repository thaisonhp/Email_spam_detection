import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from src.data_preprocessing import DataPreprocessing
from sklearn.metrics import classification_report, accuracy_score
class ModelTrainer:
    def __init__(self, model=MultinomialNB(alpha=0.25)):
        self.processor = DataPreprocessing()
        self.model = model

    def train(self, raw_data):
        # Tiền xử lý dữ liệu
        null_cols, valid_cols = self.processor.separate_null_columns(raw_data)
        cleaned_data = self.processor.clean_text(valid_cols, text_column="v2")
        encoded_data = self.processor.encode_labels(cleaned_data, label_column="v1")
        vectorized_data, vectorizer = self.processor.vectorize_text(encoded_data, text_column="v2")

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            vectorized_data, encoded_data["v1"], test_size=0.2, random_state=42
        )
        # chuyển nhãn về kiẻu dữ liệu int vì đầu vào của NB yêu cầu là int
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # Huấn luyện mô hình
        self.model.fit(X_train, y_train)

        # đánh giá mô hình
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Cross-validation
        scores = cross_val_score(self.model, vectorized_data, encoded_data["v1"].astype(int), cv=5, scoring='accuracy')
        print("Cross-validation scores:", scores)
        print("Mean cross-validation score:", scores.mean())

        # Grid search cho hyperparameters
        param_grid = {'alpha': [0.1, 0.25, 0.5, 0.75, 1]}
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Best parameters:", grid.best_params_)

        # Lưu mô hình và vectorizer
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
        joblib.dump(self.model, 'models/naive_bayes_model.joblib')
