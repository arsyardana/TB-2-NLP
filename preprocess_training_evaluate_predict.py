import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class FootballCommentSentimentAnalysis:
    def __init__(self, dataset_path):
        # Inisialisasi variabel
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = MultinomialNB()
        self.label_encoder = LabelEncoder()
        
        # Baca dataset
        self.load_dataset(dataset_path)
    
    def preprocess_text(self, text):
        """
        Membersihkan dan memproses teks
        """
        # Konversi ke huruf kecil
        text = str(text).lower()
        
        # Hapus karakter khusus dan angka
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Hapus username
        text = re.sub(r'<.*?>', '', text)
        
        # Tokenisasi
        tokens = word_tokenize(text)
        
        # Hapus stopwords
        stop_words = set(stopwords.words('indonesian'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Gabungkan kembali
        return ' '.join(tokens)
    
    def load_dataset(self, path):
        """
        Memuat dataset dan melakukan preprocessing
        """
        # Baca dataset CSV
        self.df = pd.read_csv(path)
        
        # Terapkan preprocessing
        self.df['preprocessed_text'] = self.df['komentar'].apply(self.preprocess_text)
        
        # Encode label sentimen
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Mempersiapkan data untuk training
        """
        # Vectorisasi teks
        X = self.vectorizer.fit_transform(self.df['preprocessed_text'])
        y = self.df['label_encoded']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    def train_model(self):
        """
        Melatih model klasifikasi
        """
        if self.X_train is None:
            raise ValueError("Data belum dipersiapkan. Jalankan prepare_data() terlebih dahulu.")
        
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Mengevaluasi performa model
        """
        # Prediksi
        y_pred = self.model.predict(self.X_test)
        
        # Cetak hasil evaluasi
        print("Akurasi Model:", accuracy_score(self.y_test, y_pred))
        print("\nLaporan Klasifikasi:")
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Visualisasi Confusion Matrix
        self.plot_confusion_matrix(cm, self.label_encoder.classes_)
    
    def plot_confusion_matrix(self, cm, classes):
        """
        Membuat visualisasi Confusion Matrix
        """
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, 
                    yticklabels=classes)
        plt.title('Confusion Matrix - Football Comment Sentiment Analysis')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def save_cleaned_text(self, output_path):
        """
        Menyimpan teks yang telah dibersihkan ke file CSV
        """
        if self.df is not None:
            self.df[['komentar', 'preprocessed_text', 'label']].to_csv(output_path, index=False)
            print(f"File teks yang telah dibersihkan disimpan di: {output_path}")
        else:
            print("Dataset belum dimuat.")
   
    def predict_sentiment(self, text):
        """
        Memprediksi sentimen untuk teks tunggal
        """
        # Preprocessing
        preprocessed_text = self.preprocess_text(text)
        
        # Vectorisasi
        text_vectorized = self.vectorizer.transform([preprocessed_text])
        
        # Prediksi
        prediction = self.model.predict(text_vectorized)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        
        return sentiment

def main():
    # Path dataset
    dataset_path = 'sentimen_pelatih_sepakbola.csv'
    output_cleaned_path = 'cleaned_text_output.csv'

    # Inisialisasi analisis sentimen
    sentiment_analyzer = FootballCommentSentimentAnalysis(dataset_path)
    
    # Persiapan data
    sentiment_analyzer.prepare_data()
    
    # Latih model
    sentiment_analyzer.train_model()
    
    # Evaluasi model
    sentiment_analyzer.evaluate_model()
    
    # Simpan teks yang telah dibersihkan ke file CSV
    sentiment_analyzer.save_cleaned_text(output_cleaned_path)
    
    # Contoh prediksi
    contoh_komentar = [
        "Pertahanan tim lemah sekali, pelatih tidak kompeten",
        "Strategi yang luar biasa, pertandingan mengesankan",
    ]
    
    print("\nPrediksi Sentimen:")
    for komentar in contoh_komentar:
        sentimen = sentiment_analyzer.predict_sentiment(komentar)
        print(f"Komentar: {komentar}")
        print(f"Sentimen: {sentimen}\n")

if __name__ == "__main__":
    main()