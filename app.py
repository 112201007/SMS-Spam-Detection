import nltk
import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained vectorizer and model
vectorizer = pickle.load(open("trainModel/vectorizer.pkl", 'rb'))
model = pickle.load(open("trainModel/model.pkl", 'rb'))

# Streamlit app layout
st.set_page_config(page_title="SMS Spam Detector", layout="centered", initial_sidebar_state="expanded")

# Main title and description
st.title("📩 SMS Spam Detector")
st.markdown("""
Welcome to the **SMS Spam Detector**!  
This app uses a machine learning model to classify your SMS messages as **Spam** or **Not Spam**.
""")

# Input area
st.subheader("🚀 Check your SMS message")
input_sms = st.text_area("Type or paste your SMS below:", height=150, placeholder="Enter your SMS here...")

# Button to predict
if st.button('🔍 Predict'):

    # Validate input
    if input_sms.strip() == "":
        st.error("Please enter a valid SMS message to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = vectorizer.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        st.subheader("Prediction Result")
        if result == 1:
            st.error("🚨 **This message is classified as SPAM.** 🚨")
        else:
            st.success("✅ **This message is NOT SPAM.** ✅")

# Sidebar content
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
- **App Version:** 1.0  
- **Developer:** Shreya
- **Technologies Used:** Streamlit, Scikit-learn, NLP
""")
st.sidebar.markdown("---")
st.sidebar.title("📂 Instructions")
st.sidebar.write("""
1. Enter your SMS in the input box.  
2. Click on the "Detect Spam" button.  
3. View the classification result.
""")
