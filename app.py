import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import nbimporter
import sms_detection
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tk = pickle.load(open("vectorizer.pkl", 'rb'))

model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*Created by Shyam Tripathi*")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    transformed_sms = sms_detection.transform_text(input_sms)
    vector_input = tk.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


