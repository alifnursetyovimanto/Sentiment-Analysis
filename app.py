import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# ── Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    .stTextArea textarea { font-size: 15px; }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .positive { background-color: #E1F5EE; color: #085041; }
    .negative { background-color: #FAECE7; color: #712B13; }
    .neutral  { background-color: #EEEDFE; color: #3C3489; }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model
@st.cache_resource
def load_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return tfidf, lr, le


# ── Text preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)


# ── Predict function
def predict(text, tfidf, lr, le):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred_class = lr.predict(vec)[0]
    pred_prob = lr.predict_proba(vec)[0]
    sentiment = le.inverse_transform([pred_class])[0]
    confidence = pred_prob[pred_class]
    probs = {le.classes_[i]: pred_prob[i] for i in range(len(le.classes_))}
    return sentiment, confidence, probs, cleaned


# ── Emoji & color mapping
def get_emoji(sentiment):
    return {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(sentiment, '🤔')

def get_color(sentiment):
    return {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}.get(sentiment, 'neutral')


# ── Main App
def main():
    # Header
    st.title("💬 Social Media Sentiment Analysis")
    st.markdown("""
    Classify social media text into **Positive**, **Neutral**, or **Negative** sentiment
    using a Logistic Regression model trained on 499 social media posts.
    """)

    st.divider()

    # Load model
    try:
        tfidf, lr, le = load_model()
        model_loaded = True
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first to generate model files.")
        model_loaded = False
        st.stop()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📋 Batch Prediction", "📊 Model Info"])

    # ── Tab 1: Single Prediction
    with tab1:
        st.subheader("Analyze a single text")
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste social media text here...",
            height=120
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button("Analyze Sentiment", use_container_width=True, type="primary")

        if predict_btn and text_input.strip():
            sentiment, confidence, probs, cleaned = predict(text_input, tfidf, lr, le)
            emoji = get_emoji(sentiment)
            css_class = get_color(sentiment)

            # Result box
            st.markdown(f"""
            <div class="result-box {css_class}">
                <h1>{emoji}</h1>
                <h2>{sentiment.upper()}</h2>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Probability distribution:**")
            for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                emoji_label = get_emoji(label)
                st.progress(float(prob), text=f"{emoji_label} {label.capitalize()}: {prob:.1%}")

            # Cleaned text
            with st.expander("See preprocessed text"):
                st.code(cleaned)

        elif predict_btn and not text_input.strip():
            st.warning("Please enter some text to analyze.")

        # Example texts
        st.markdown("**Try these examples:**")
        examples = [
            "This product is absolutely amazing, I love it!",
            "The service was okay, nothing special.",
            "Terrible experience, I'm very disappointed with the quality.",
            "Just received my order today.",
            "Best purchase I've made this year, highly recommend!"
        ]

        for example in examples:
            if st.button(f"📝 {example[:50]}...", key=example):
                sentiment, confidence, probs, cleaned = predict(example, tfidf, lr, le)
                emoji = get_emoji(sentiment)
                css_class = get_color(sentiment)
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <h3>{emoji} {sentiment.upper()}</h3>
                    <p>Confidence: <strong>{confidence:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: Batch Prediction
    with tab2:
        st.subheader("Analyze multiple texts at once")
        st.markdown("Upload a CSV file with a column named `text`")

        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV must have a column named 'text'")
                else:
                    st.success(f"Loaded {len(df)} rows")
                    st.dataframe(df.head(), use_container_width=True)

                    if st.button("Run Batch Analysis", type="primary"):
                        with st.spinner("Analyzing..."):
                            results = []
                            for text in df['text']:
                                sentiment, confidence, probs, _ = predict(
                                    str(text), tfidf, lr, le
                                )
                                results.append({
                                    'text': text,
                                    'sentiment': sentiment,
                                    'confidence': f"{confidence:.1%}"
                                })

                            result_df = pd.DataFrame(results)
                            st.dataframe(result_df, use_container_width=True)

                            # Summary chart
                            st.markdown("**Sentiment distribution:**")
                            counts = result_df['sentiment'].value_counts()
                            fig, ax = plt.subplots(figsize=(6, 3))
                            colors = {'positive': '#1D9E75',
                                      'neutral': '#534AB7',
                                      'negative': '#D85A30'}
                            bar_colors = [colors.get(s, '#888') for s in counts.index]
                            ax.bar(counts.index, counts.values, color=bar_colors)
                            ax.set_ylabel('Count')
                            ax.set_title('Sentiment Distribution')
                            st.pyplot(fig)
                            plt.close()

                            # Download
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "Download Results",
                                csv,
                                "sentiment_results.csv",
                                "text/csv"
                            )

            except Exception as e:
                st.error(f"Error: {e}")

    # ── Tab 3: Model Info
    with tab3:
        st.subheader("About this model")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Logistic Regression")
        with col2:
            st.metric("Accuracy", "71%")
        with col3:
            st.metric("F1-Score (macro)", "0.69")

        st.markdown("""
        ### Why Logistic Regression?

        This project compared **Bidirectional LSTM** vs **Logistic Regression**
        on 499 social media texts. Key finding:

        | Model | Accuracy | F1-Score |
        |---|---|---|
        | Logistic Regression | **71%** | **0.69** |
        | Bidirectional LSTM | 69% | 0.66 |

        Logistic Regression outperformed LSTM because the dataset (499 samples)
        is too small for deep learning to generalize well. This demonstrates
        that **model complexity must match data size**.

        ### Dataset
        - 499 social media posts (Twitter & Facebook)
        - 3 classes: Positive (166), Neutral (199), Negative (134)
        - Language: English

        ### Preprocessing pipeline
        1. Lowercase conversion
        2. Remove URLs, mentions, hashtags
        3. Remove non-alphabetic characters
        4. Remove stopwords (NLTK)
        5. TF-IDF vectorization (max 5,000 features, bigrams)

        ### Limitations
        - Small dataset (499 samples) — accuracy may vary on unseen domains
        - English only — not suitable for Indonesian text
        - Does not handle sarcasm well

        ### Built by
        **Alif Nursetyo Vimanto** · [GitHub](https://github.com/alifnursetyovimanto)
        · [LinkedIn](https://linkedin.com/in/alifnursetyo)
        """)


if __name__ == "__main__":
    main()
