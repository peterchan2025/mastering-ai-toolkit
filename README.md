🧠 Mastering the AI Toolkit
AI Tools and Applications Assignment
📘 Overview

This project demonstrates practical mastery of modern Artificial Intelligence (AI) frameworks and tools, including Scikit-learn, TensorFlow, and spaCy.

Developed under the theme “Mastering the AI Toolkit”, the assignment integrates classical machine learning, deep learning, and natural language processing (NLP) techniques while emphasizing ethical AI practices and optimization strategies.

The work was completed collaboratively to enhance teamwork and real-world AI engineering skills.

⚙️ Setup & Installation
1️⃣ Clone the Repository
git clone 
cd mastering-ai-toolkit

2️⃣ Install Dependencies
pip install -r requirements.txt


If running in Google Colab, also execute:

!pip install spacy textblob tensorflow scikit-learn matplotlib
!python -m spacy download en_core_web_sm

3️⃣ Run the Notebook

Open and execute:

notebooks/Part2_Practical.ipynb

🚀 Practical Implementation
🧠 Task 1: Classical Machine Learning (Scikit-learn)

Dataset: Iris Species Dataset
Goal: Build and evaluate a Decision Tree Classifier to predict iris flower species.
Key Steps:

Data preprocessing (handling missing values, label encoding)

Model training and evaluation (accuracy, precision, recall)
Tools: pandas, scikit-learn

🔢 Task 2: Deep Learning (TensorFlow)

Dataset: MNIST Handwritten Digits
Goal: Train a Convolutional Neural Network (CNN) to classify handwritten digits.
Achievements:

Model accuracy >95% on test data

Visualization of predictions for sample images
Tools: tensorflow, keras, matplotlib

💬 Task 3: Natural Language Processing (spaCy)

Dataset: Amazon Product Reviews (sample text)
Goal: Extract product and brand names using NER, and perform sentiment analysis.
Approach:

NER: Extracted entities using spaCy

Sentiment: Rule-based analysis with TextBlob
Tools: spaCy, TextBlob

📊 Results Summary
Task	Framework	Dataset	Key Metric	Result
Task 1	Scikit-learn	Iris Dataset	Accuracy	~96%
Task 2	TensorFlow	MNIST	Test Accuracy	>95%
Task 3	spaCy + TextBlob	Amazon Reviews	Sentiment Accuracy	Positive/Negative Classification

📸 Visual outputs (graphs and NER examples) are stored in notebooks/model_outputs/.

🧠 Ethical Considerations

Potential Biases Identified:

MNIST Model: Limited handwriting style diversity may reduce generalization.

Amazon Reviews Analysis: Possible language and sentiment polarity bias (over-representation of specific products or tones).

Mitigation Strategies:

Use of TensorFlow Fairness Indicators to audit fairness metrics.

Implementation of rule-based filters in spaCy to balance sentiment weighting.

Incorporation of diverse and representative training samples.
📚 References

TensorFlow Documentation — https://www.tensorflow.org

PyTorch Documentation — https://pytorch.org

Scikit-learn Documentation — https://scikit-learn.org/stable

spaCy Documentation — https://spacy.io
