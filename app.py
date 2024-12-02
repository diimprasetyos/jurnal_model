from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import download

# Unduh stopwords
download('stopwords')

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer
model = pickle.load(open('model/sentiment_jurnal.pkl', 'rb'))
tokenizer = pickle.load(open('model/tokenizer_jurnal.pkl', 'rb'))

# Stopwords dan stemmer
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# Fungsi preprocessing teks
def preprocess_text(text):
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # Menghapus angka
    text = re.sub(r'\d+', '', text)

    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Mengubah teks ke lowercase
    text = text.lower()

    # Menghapus stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stemming
    text = stemmer.stem(text)

    return text


# Endpoint API untuk prediksi sentimen
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Ambil data teks dari request
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'Input text is required'}), 400

        # Preprocess teks
        processed_text = preprocess_text(text)

        # Tokenisasi dan padding
        seq = tokenizer.texts_to_sequences([processed_text])
        seq = pad_sequences(seq, maxlen=200, dtype='int32', value=0)

        # Prediksi sentimen
        sentiment = model.predict(seq, batch_size=1, verbose=0)[0]
        negative_prob = sentiment[0] * 100
        positive_prob = sentiment[1] * 100

        # Return hasil
        return jsonify({
            'negative': f'{negative_prob:.2f}%',
            'positive': f'{positive_prob:.2f}%',
            'sentiment': 'positive' if positive_prob > negative_prob else 'negative'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# (Opsional) Halaman UI untuk testing
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
