# SMS Spam Detection System

A machine learning-powered web application that classifies SMS messages as spam or ham (legitimate messages) using Natural Language Processing techniques.

## 🚀 Features

- **Real-time SMS Classification**: Instantly classify text messages as spam or legitimate
- **Multiple ML Models**: Choose between Naive Bayes and Support Vector Machine (SVM) classifiers
- **Interactive Web Interface**: Clean and user-friendly Streamlit-based web application
- **Text Preprocessing**: Advanced NLP preprocessing including lemmatization and stopword removal
- **Visual Feedback**: Color-coded results (red for spam, green for ham)

## 📁 Project Structure

```
├── app1.py                     # Streamlit web application
├── SMS_Spam_Detection.ipynb    # Jupyter notebook with model training
├── mnb_model.pkl              # Trained Multinomial Naive Bayes model
├── svm_model.pkl              # Trained Support Vector Machine model
├── SMSSpamCollection.txt      # Original dataset
├── SMSSpamCollection_new.txt  # Preprocessed dataset
└── README.md                  # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning models
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **Joblib** - Model serialization

## 📊 Dataset

The project uses the SMS Spam Collection dataset containing 5,574 SMS messages labeled as either:
- **Ham**: Legitimate messages
- **Spam**: Unwanted/promotional messages

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/achu1010/SMS-SPAM-DETECTOR.git
   cd SMS-SPAM-DETECTOR
   ```

2. Install required packages:
   ```bash
   pip install streamlit scikit-learn nltk pandas joblib
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🚀 Usage

### Running the Web Application

1. Navigate to the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app1.py
   ```
3. Open your browser and go to `http://localhost:8501`

### Using the Application

1. **Select a Classifier**: Choose between "Naive Bayes" or "SVM" from the dropdown
2. **Enter Message**: Type or paste the SMS message you want to classify
3. **Click Classify**: Press the "Classify" button to get results
4. **View Results**: The classification result will appear with color coding

### Training Your Own Models

Open and run the `SMS_Spam_Detection.ipynb` notebook to:
- Explore the dataset
- Preprocess the text data
- Train and evaluate different models
- Save new model files

## 🤖 Model Performance

The system includes two trained models:

1. **Multinomial Naive Bayes**: Fast and efficient for text classification
2. **Support Vector Machine**: Higher accuracy for complex patterns

Both models use TF-IDF vectorization and achieve high accuracy on the test dataset.

## 📝 Text Preprocessing

The application performs the following preprocessing steps:
- Remove special characters and numbers
- Convert to lowercase
- Tokenization
- Remove English stopwords
- Lemmatization using WordNet

## 🎨 User Interface

The Streamlit interface features:
- Professional styling with custom CSS
- Responsive design
- Clear visual feedback
- Easy model selection
- Real-time classification

## 🔮 Future Enhancements

- [ ] Add more ML models (Random Forest, Neural Networks)
- [ ] Implement model performance metrics display
- [ ] Add batch processing for multiple messages
- [ ] Include confidence scores
- [ ] Deploy to cloud platforms (Heroku, AWS, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**achu1010**
- GitHub: [@achu1010](https://github.com/achu1010)

## 🙏 Acknowledgments

- SMS Spam Collection dataset from UCI Machine Learning Repository
- Streamlit community for the excellent web framework
- scikit-learn developers for the machine learning tools

---

⭐ If you found this project helpful, please give it a star!
