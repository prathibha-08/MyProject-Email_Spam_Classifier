import pandas as pd
import string
import matplotlib.pyplot as plt
from tkinter import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

df['message'] = df['message'].apply(clean_text)

# ---------------- VECTORIZATION ----------------
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- MODEL ----------------
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- GUI FUNCTIONS ----------------
def check_spam():
    msg = entry.get()
    if msg.strip() == "":
        output.config(text="Please enter a message", fg="orange")
        return

    msg_clean = clean_text(msg)
    msg_vector = vectorizer.transform([msg_clean])
    result = model.predict(msg_vector)

    if result[0] == 1:
        output.config(text="SPAM ❌", fg="red")
    else:
        output.config(text="NOT SPAM ✅", fg="green")

def show_charts():
    # Bar Chart
    counts = df['label'].value_counts()

    plt.figure()
    plt.bar(['Ham', 'Spam'], counts)
    plt.title("Spam vs Ham Message Count")
    plt.xlabel("Message Type")
    plt.ylabel("Number of Messages")
    plt.show()

    # Pie Chart
    error = 1 - accuracy

    plt.figure()
    plt.pie(
        [accuracy, error],
        labels=['Correct Predictions', 'Wrong Predictions'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Model Accuracy")
    plt.show()

# ---------------- GUI WINDOW ----------------
root = Tk()
root.title("Email Spam Classifier")
root.geometry("420x300")

Label(root, text="Email Spam Classifier", font=("Arial", 16, "bold")).pack(pady=10)

Label(root, text="Enter Email Message:", font=("Arial", 12)).pack(pady=5)
entry = Entry(root, width=50)
entry.pack(pady=5)

Button(root, text="Check Spam", command=check_spam, width=20).pack(pady=10)
Button(root, text="Show Charts", command=show_charts, width=20).pack(pady=5)

output = Label(root, text="", font=("Arial", 14))
output.pack(pady=10)

root.mainloop()
