# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

import nltk

# Specify the path to your NLTK data directory
nltk_data_path = "stopwords"

# Append the path to nltk.data.path list
nltk.data.path.append(nltk_data_path)

# Load and preprocess data
data = pd.read_csv("sentiment_data.csv")  # Assuming you have a CSV file with text and corresponding labels
data = data.dropna()  # Remove any rows with missing values
data["text"] = data["text"].apply(lambda x: re.sub(r'\W', ' ', x))  # Remove special characters
data["text"] = data["text"].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))  # Remove single characters
data["text"] = data["text"].apply(lambda x: re.sub(r'\s+', ' ', x))  # Remove extra spaces
data["text"] = data["text"].apply(lambda x: x.lower())  # Convert to lowercase

# Tokenization, Lemmatization, and stop words removal
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# data["text"] = data["text"].apply(lambda x: word_tokenize(x))
# data["text"] = data["text"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x if word not in stop_words])
stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
data["text"] = data["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(data["text"].apply(lambda x: ' '.join(x)))
y = data["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize LinearSVC classifier
classifier = LinearSVC()

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
