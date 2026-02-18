📧 Email Spam Classifier

A Machine Learning-based Email Spam Classifier built using Python, Scikit-learn, Pandas, and NLTK. This project classifies email messages as Spam or Ham (Not Spam) using the Multinomial Naive Bayes algorithm and includes a simple Tkinter GUI for user interaction.

🚀 Features

Text preprocessing (lowercase conversion, punctuation removal, stopword removal)

Feature extraction using CountVectorizer

Spam classification using Multinomial Naive Bayes

Simple and interactive Tkinter GUI

Fast and accurate predictions

🧠 Technologies Used

Python

Pandas

NLTK

Scikit-learn

Tkinter

📂 Project Structure
Email-Spam-Classifier/
│
├── spam.csv          # Dataset
├── app.py            # Main application file
├── README.md         # Project documentation
└── requirements.txt  # Required libraries

⚙️ How It Works

Load and clean the dataset (spam.csv).

Preprocess text using NLP techniques.

Convert text into numerical vectors using CountVectorizer.

Train the model using Multinomial Naive Bayes.

Use the Tkinter GUI to input messages and get predictions.

▶️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/email-spam-classifier.git


Navigate to the project folder:

cd email-spam-classifier


Install required libraries:

pip install -r requirements.txt


Run the application:

python app.py

🎯 Objective

The goal of this project is to demonstrate how Machine Learning and Natural Language Processing (NLP) can be used to build a practical spam detection system.

🔮 Future Improvements

Add probability/confidence score

Improve GUI design

Deploy as a web application

Add real-time email filtering
