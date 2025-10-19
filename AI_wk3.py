# =========================================================
# AI Tools Assignment - "Mastering the AI Toolkit"
# Author(s): Your Group Names
# Frameworks Used: Scikit-learn, TensorFlow, spaCy
# =========================================================

# ---------------------------------------------------------
# Task 1: Classical ML with Scikit-learn
# Dataset: Iris Dataset
# Goal: Train a Decision Tree Classifier
# ---------------------------------------------------------

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Simulate missing values and handle them
df.iloc[2, 1] = None
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode labels
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# Split data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print("=== Decision Tree Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# Task 2: Deep Learning with TensorFlow
# Dataset: MNIST Handwritten Digits
# Goal: Build CNN model (>95% test accuracy)
# ---------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\n=== CNN Model Evaluation ===")
print("Test Accuracy:", test_acc)

# Visualize predictions
predictions = model.predict(X_test[:5])
plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.show()

# ---------------------------------------------------------
# Task 3: NLP with spaCy
# Dataset: Sample Amazon Product Reviews
# Goal: Perform NER and sentiment analysis
# ---------------------------------------------------------

import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Example text data
reviews = [
    "I love the new Apple iPhone! The camera quality is amazing.",
    "The Samsung Galaxy is too expensive for the features it offers.",
    "I bought a Dell laptop and it works perfectly.",
]

print("\n=== Named Entity Recognition & Sentiment Analysis ===")
for text in reviews:
    doc = nlp(text)
    print(f"\nReview: {text}")
    print("Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    sentiment = TextBlob(text).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print("Sentiment:", sentiment_label)

