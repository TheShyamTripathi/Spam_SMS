import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform the text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove special characters and numbers
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Load the trained vectorizer and model
try:
    vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError as e:
    st.error("Required files (vectorizer.pkl, model.pkl) are missing. Please ensure they are in the same directory.")
    st.stop()

# Streamlit app interface
st.title("SMS Spam Detection Model")
st.write("*Created by Shyam Tripathi*")

# Input for SMS text
input_sms = st.text_area("Enter the SMS", height=100)

# Predict button
if st.button('Predict'):
    # Check if input is valid
    if not input_sms.strip():
        st.error("Please enter a valid SMS!")
    else:
        # Preprocess the input SMS
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the input
        vector_input = vectorizer.transform([transformed_sms])
        
        # Predict using the model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
