import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # Fixed typo
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
st.set_page_config(page_title="Review Sentiment Analysis", layout="centered", initial_sidebar_state="collapsed")


model = load_model('model.h5')

word_index = imdb.get_word_index()

def preprocesstext(review):
    words = review.lower().split()
    enc_rev = [word_index.get(word,2)+3 for word in words]
    padd = sequence.pad_sequences([enc_rev],maxlen=500)
    return padd


def predict_setiment(review):
    review = preprocesstext(review=review)
    pred = model.predict(review)
    sentiment = 'Positive' if pred[0][0] >0.5 else 'Negative'
    return sentiment,pred[0][0]
from PIL import Image
header_image = Image.open("image.png")
bad = Image.open("bad.png")
good = Image.open('images.png')
# Configuration


# UI Design
st.title("📊 Review Sentiment Analysis App")
st.image(header_image)

st.markdown(
    """
    Welcome to the **Review Sentiment Analysis App**!  
    Enter a review below, and the model will predict whether the sentiment is **Positive** or **Negative**.  
    """
)

# Input Section
st.markdown("### 📝 Enter Your Review:")
user_input = st.text_area("Type your review here...", placeholder="E.g., The Movie is amazing!", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing!")
    else:
        sent, score = predict_setiment(user_input)
        st.subheader("Prediction Result:")
        if(sent== 'Negative'):
            st.markdown(
            f"""
            - **Sentiment:** 🎉 `{sent}`  
            - **Confidence:** `{(1-score)*100:.2f}%`  
            """
        )
            st.image(bad)
        else:
            st.markdown(
            f"""
            - **Sentiment:** 🎉 `{sent}`  
            - **Confidence:** `{score*100:.2f}%`  
            """
        )
            st.image(good)


# Footer Section

st.markdown(
    """
    ---  
    Made with ❤️ using **Streamlit** and **TensorFlow**.  
    """
)

st.markdown('''
    linkedin: TilakAkasapu
''')
