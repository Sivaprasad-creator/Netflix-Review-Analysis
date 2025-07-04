import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Page config MUST be first Streamlit command
st.set_page_config(page_title="Netflix Review Sentiment", layout="centered")

# ✅ Optional background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://variety.com/wp-content/uploads/2020/05/netflix-logo.png");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ✅ Load model artifacts
model_path = 'C:/Users/Acer/Downloads/Python/Github/Netflix Review/'
model = joblib.load(model_path + 'stacking_sentiment_model.pkl')
vectorizer = joblib.load(model_path + 'tfidf_vectorizer.pkl')
label_encoder = joblib.load(model_path + 'label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ✅ App Title
st.title("🎬 Netflix Review Sentiment Predictor")

# --- Reset logic ---
if "clear_text" not in st.session_state:
    st.session_state.clear_text = False

if st.session_state.clear_text:
    st.session_state.review_input = ""
    st.session_state.clear_text = False
    st.rerun()

# Text input
review = st.text_area("📝 Enter your Netflix review:", key="review_input")

# Predict
if st.button("🔍 Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)
        label = pred[0]

        emoji_map = {
            'Positive': '😊',
            'Neutral': '😐',
            'Negative': '😡'
        }
        st.success(f"**Sentiment:** {label} {emoji_map[label]}")

# Reset
if st.button("🔄 Reset"):
    st.session_state.clear_text = True
    st.rerun()
