import streamlit as st
import joblib
import os
from utils import transform_text # Essential import

st.set_page_config(page_title="Pro Spam Detector", page_icon="🛡️")

# Custom CSS for better look
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    .stButton button { width: 100%; background-color: #4CAF50; color: white; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Check if advanced model exists, otherwise fall back or warn
    if os.path.exists('models/spam_model_advanced.pkl'):
        return joblib.load('models/spam_model_advanced.pkl')
    else:
        return None

st.title("🛡️ Pro-Level Email Spam Classifier")
st.markdown("Using **Support Vector Machines (SVM)** and **NLP Stemming** for high-precision detection.")

user_input = st.text_area("Paste Email Text Here", height=200)

if st.button("Analyze Email"):
    model = load_model()
    
    if model is None:
        st.error("Model not found! Please run 'train_advanced.py' first.")
    elif not user_input.strip():
        st.warning("Please enter text.")
    else:
        # 1. Preprocess
        transformed_email = transform_text(user_input)
        
        # 2. Predict
        # Note: The vectorizer is inside the pipeline, so we just pass the text
        prediction = model.predict([transformed_email])[0]
        proba = model.predict_proba([transformed_email])[0]

        # 3. Display
        if prediction == 1:
            confidence = proba[1] * 100
            st.error(f"🚨 SPAM DETECTED")
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            st.write("Analysis: High probability of unsolicited content.")
        else:
            confidence = proba[0] * 100
            st.success(f"✅ NOT SPAM (Ham)")
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            st.write("Analysis: This message appears to be legitimate.")

# Sidebar
st.sidebar.header("How it works")
st.sidebar.write("""
This model performs 3 steps:
1. **Cleaning:** Removes stop words ('the', 'is') and reduces words to roots (Stemming).
2. **Vectorization:** Converts text to math using TF-IDF (Top 3000 words).
3. **Classification:** Uses an SVM with a Sigmoid Kernel to draw a decision boundary.
""")