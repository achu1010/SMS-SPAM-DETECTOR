import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
nb_model = joblib.load('mnb_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Preprocess function
def preprocess_data(text):
    lm = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower().split()
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    return " ".join(review)

# Streamlit Layout
st.title("Spam/Ham Message Classifier")

# Dropdown for classifier selection
classifier_choice = st.selectbox("Choose a classifier", ["Naive Bayes", "SVM"])

# Text area for input
user_input = st.text_area("Enter your message here")

# Classify button
if st.button("Classify"):
    processed_text = preprocess_data(user_input)
    prediction = nb_model.predict([processed_text])[0] if classifier_choice == "Naive Bayes" else svm_model.predict([processed_text])[0]
    
    # Display results
    st.markdown(
        f"<h2 style='color: {'red' if prediction == 'spam' else 'green'};'>This message is classified as: {'Spam' if prediction == 'spam' else 'Ham'}</h2>",
        unsafe_allow_html=True
    )

# Professional Style Adjustments
st.markdown(
    """
    <style>
    .stTextInput > label {
        color: #2c3e50;  /* Dark Blue */
        font-weight: 600;
        font-size: 18px;
    }
    .stButton > button {
        background-color: #3498db; /* Light Blue */
        color: white;
        padding: 12px 28px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2980b9; /* Darker Blue */
    }
    .stSelectbox > label {
        font-weight: 600;
        font-size: 16px;
        color: #34495e; /* Gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)
