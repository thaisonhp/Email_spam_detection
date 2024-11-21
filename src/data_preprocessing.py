import spacy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer

class DataPreprocessing:
    def __init__(self):
        self.vectorizer = None  # Vectorizer cho Tfidf
        self.label_encoder = None  # Bộ mã hóa nhãn
        self.nlp = spacy.load("en_core_web_sm")  # Tải spaCy model (có thể dùng model khác nếu cần)

    def check_info(self, data: pd.DataFrame):
        """In thông tin của dataframe."""
        print(data.info())

    def separate_null_columns(self, data: pd.DataFrame):
        """Tách các cột có giá trị null và không có giá trị null."""
        null_columns = data.loc[:, data.isnull().any()]
        non_null_columns = data.loc[:, ~data.isnull().any()]
        return null_columns, non_null_columns

    def clean_text_with_spacy(self, text: str) -> str:
        """
        Làm sạch văn bản sử dụng spaCy:
        - Loại bỏ stop words
        - Chỉ giữ lại từ dạng gốc (lemmatization)
        - Loại bỏ ký tự không phải chữ
        """
        doc = self.nlp(text.lower())  # Chuyển về chữ thường và parse văn bản
        # Sử dụng tokenizer của spaCy và xử lý văn bản
        tokens = [token.lemma_ for token in doc if
                  not token.is_stop and token.is_alpha]  # Lemmatization và lọc stop words
        return " ".join(tokens)  # Ghép các token thành chuỗi



    def clean_text(self, data: pd.DataFrame, text_column: str):
        """Làm sạch văn bản trong cột sử dụng spaCy."""
        # data[text_column] = data[text_column].apply(self.clean_text_with_spacy)
        data.loc[:, text_column] = data[text_column].apply(self.clean_text_with_spacy) # làm như này để tránh ảnh hưởng view của dữ liệu

        return data


    def encode_labels(self, data: pd.DataFrame, label_column: str):
        """Mã hóa nhãn từ dạng chuỗi sang số."""
        self.label_encoder = LabelEncoder()
        # data[label_column] = self.label_encoder.fit_transform(data[label_column])
        data.loc[:, label_column] = self.label_encoder.fit_transform(data[label_column])

        return data

    def vectorize_text(self, data: pd.DataFrame, text_column: str):
        """Vector hóa cột văn bản sử dụng TfidfVectorizer."""
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(data[text_column]).toarray()
        return X , self.vectorizer
