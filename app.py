import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Tạo instance FastAPI
app = FastAPI()

# Tải mô hình và vectorizer đã lưu
model = joblib.load('models/naive_bayes_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')  # Tải vectorizer đã huấn luyện

# Tạo một lớp để nhận dữ liệu đầu vào
class TextInput(BaseModel):
    text: str

# API endpoint để dự đoán
@app.post("/predict/")
def predict(input_data: TextInput):
    # Tiền xử lý văn bản: Vectorize văn bản với TfidfVectorizer đã được huấn luyện
    vectorized_input = vectorizer.transform([input_data.text])
    print(vectorized_input)  # In ra các đặc trưng vectorized
    # Dự đoán với mô hình Naive Bayes
    prediction = model.predict(vectorized_input)

    # Trả về kết quả dự đoán
    return {"prediction": int(prediction[0])}
