
# **SMS Spam Detection Project**

## **Overview**
This project is part of a 1-month AI internship under the **AICTE-Internship** program by Microsoft & SAP through **TechSaksham**. The goal of this project is to develop a **Spam SMS Detection Application** that identifies and classifies SMS messages as spam or not spam using a machine learning model.

---

## **Deployed Application**
Access the deployed application here:  
👉 [SMS Spam Detection App](https://shyamtripathi-spam-sms.streamlit.app/)

---

## **Project Structure**
### **1. Model Creation**
File: `sms_detection.ipynb`  
This Jupyter notebook handles:
- Data preprocessing: Tokenization, stemming, and removal of stopwords.
- Feature extraction using `TfidfVectorizer`.
- Model training with a **Naive Bayes classifier**.
- Model evaluation using accuracy, precision, and confusion matrix.

### **2. Frontend and Deployment**
File: `app.py`  
This Python file:
- Implements the web interface using **Streamlit**.
- Accepts SMS input from the user.
- Uses the trained model to classify the SMS as spam or not spam.

---

## **Key Features**
1. **Interactive Web Interface**:
   - User-friendly interface to enter SMS text.
   - Displays results with clear labels (`Spam` or `Not Spam`).

2. **Machine Learning Model**:
   - Uses a **Naive Bayes Classifier** for spam detection.
   - Processes SMS text with `nltk` for text cleaning and stemming.

3. **Deployment**:
   - Hosted on **Streamlit Community Cloud**.
   - Accessible via the provided URL for public use.

---

## **Timeline**
This project was developed over a 4-week internship program:
- **Week 1**: Orientation, problem statement definition, and project allocation.
- **Week 2**: Literature survey and dataset analysis.
- **Week 3**: Model development and testing.
- **Week 4**: Application deployment and final report preparation.

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**: `nltk`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`, `wordcloud`
- **Deployment**: Streamlit Community Cloud

---

## **How to Run Locally**
### Prerequisites:
1. Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Steps:
1. Clone this repository.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL provided by Streamlit (e.g., `http://localhost:8501`).

---

## **Acknowledgments**
This project is part of the **AICTE Internship Program**:
- Organized by **Microsoft**, **SAP**, and **Edunet Foundation**.
- Focused on **AI: Transformative Learning with TechSaksham**.

Special thanks to the mentors and organizers for their guidance and support.

