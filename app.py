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
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="centered")
# Adding an image or logo at the top-left corner
st.image("edunet.png")
# Adding an image or logo at the top-left corner
st.image("./TechSaksham.jpg", width=250)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>SMS Spam Detection Model</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #2196F3;'>Created by Shyam Tripathi</h3>", unsafe_allow_html=True)

# Adding an image or logo at the top-left corner
st.image("./S#S.png", width=100)  # Replace with your image URL

# Adding a second logo at the top-right corner using custom CSS
st.markdown("""
    <style>
        .logo-top-right {
            position: fixed;
            top: 10px;
            right: 10px;
            width: 100px;  /* Adjust the size as needed */
            z-index: 1000;
        }
            
        .logo-left-right {
            position: fixed;
            top: 10px;
            left: 10px;
            width: 100px;  /* Adjust the size as needed */
            z-index: 1000;
        }
    </style>
    <img class="logo-top-right" src="./edunet.png" alt="Logo">  
    <img class="logo-left-right" src="./TechSaksham.jpg" alt="Logo">
""", unsafe_allow_html=True)

# Input for SMS text with styling
input_sms = st.text_area("Enter the SMS below:", height=100, placeholder="Type the SMS here...", max_chars=500)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Predict button with custom style
if st.button('🔍 Predict', key="predict_button", help="Click to predict if the SMS is Spam or Not Spam"):
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
        
        # Display the result with enhanced styling
        if result == 1:
            st.markdown("<h3 style='text-align: center; color: red;'>🚨 Spam Message Detected!</h3>", unsafe_allow_html=True)
            st.balloons()  # Optional: Adds a balloon effect to indicate spam
        else:
            st.markdown("<h3 style='text-align: center; color: green;'>✅ Not Spam</h3>", unsafe_allow_html=True)

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Footer section
st.markdown("<footer style='text-align: center; font-size: 22px; color: #222;'>Built with ❤️ by Shyam Tripathi</footer>", unsafe_allow_html=True)
