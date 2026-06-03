# Social Media Sentiment Analysis

Sentiment classification web app for social media text — comparing Bidirectional LSTM vs Logistic Regression, deployed with Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f)

---

## Live Demo

🔗 **[Try the app on Streamlit Cloud](#)** ← ganti dengan link lo

---

## Hasil singkat

| Model | Accuracy | F1-Score (macro) |
|---|---|---|
| **Logistic Regression** | **71%** | **0.69** |
| Bidirectional LSTM | 69% | 0.66 |

**Key finding:** Logistic Regression outperforms LSTM on this dataset —
proving that model complexity must match data size, not the other way around.

---

## Features

- **Single prediction** — analyze one text with confidence score
- **Batch prediction** — upload CSV and analyze multiple texts at once
- **Probability distribution** — see confidence across all 3 classes
- **Model info** — transparent documentation of methodology and limitations

---

## Struktur project

```
sentiment-analysis/
├── app.py                        # Streamlit web application
├── social_media_sentiment.ipynb  # Training notebook (full analysis)
├── sentiment_analysis.csv        # Dataset (499 social media posts)
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── lr_model.pkl                  # Saved Logistic Regression model
├── label_encoder.pkl             # Saved Label Encoder
├── requirements.txt              # Dependencies
└── assets/
    ├── eda_distribution.png
    ├── training_curve.png
    ├── confusion_matrix_comparison.png
    └── model_comparison.png
```

---

## Cara jalankan lokal

```bash
# 1. Clone repo
git clone https://github.com/alifnursetyovimanto/sentiment-analysis.git
cd sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan app
streamlit run app.py
```

---

## Cara deploy ke Streamlit Cloud

1. Push semua file ke GitHub (termasuk `.pkl` files)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` sebagai main file → Deploy
4. App live dalam 2–3 menit

---

## Pendekatan & keputusan

### Masalah bisnis
Platform e-commerce dan media sosial menerima ribuan komentar setiap hari.
Mengklasifikasikan sentimen secara manual tidak scalable — dibutuhkan model
otomatis untuk memahami opini pelanggan secara real-time.

### Dataset
- 499 social media posts (Twitter & Facebook)
- 3 kelas: Positive (166), Neutral (199), Negative (134)
- Relatif balanced — tidak perlu resampling

### Kenapa membandingkan dua model?
Best practice di industri adalah membangun baseline sederhana sebelum
model kompleks. Kalau baseline sudah cukup baik, tidak perlu deep learning.

### Temuan utama
Logistic Regression (71%) mengungguli Bidirectional LSTM (69%). LSTM
mengalami overfitting karena dataset terlalu kecil (499 sampel) untuk
deep learning yang robust — training accuracy mencapai 95% sementara
validation stagnan di 69%.

### Keputusan deployment
Logistic Regression dipilih sebagai production model karena:
- Akurasi lebih tinggi
- Inferensi lebih cepat
- Tidak membutuhkan GPU
- Lebih mudah di-maintain

### Keterbatasan
- Dataset kecil (499 sampel) — accuracy bisa berbeda di domain lain
- English only — tidak akurat untuk teks Indonesia
- Tidak menangani sarkasme dengan baik
- Next step: fine-tune BERT/RoBERTa dengan dataset lebih besar

---

## Tech stack

- **Web app:** Streamlit
- **ML:** Scikit-learn (Logistic Regression, TF-IDF)
- **Deep learning:** TensorFlow/Keras (Bidirectional LSTM)
- **NLP:** NLTK (preprocessing, stopwords)
- **Visualization:** Matplotlib, Seaborn

---

*Project ini dibuat sebagai bagian dari portofolio data scientist.
Feedback dan pertanyaan sangat disambut — silakan buka issue atau
hubungi saya di [LinkedIn](https://linkedin.com/in/alifnursetyo).*
