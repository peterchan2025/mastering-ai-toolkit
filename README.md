
# Mastering the AI Toolkit  
**AI Tools and Applications Assignment**  
> A practical project showcasing applied Artificial Intelligence using Scikit-learn, TensorFlow, and spaCy.  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Toolkit-yellow.svg)](https://scikit-learn.org/stable/)  
[![spaCy](https://img.shields.io/badge/spaCy-NLP-green.svg)](https://spacy.io/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)  

---

## 📘 Overview  
This project demonstrates practical mastery of modern Artificial Intelligence (AI) frameworks and tools, including **Scikit-learn**, **TensorFlow**, and **spaCy**.  

Developed under the theme **“Mastering the AI Toolkit”**, the assignment integrates **classical machine learning**, **deep learning**, and **natural language processing (NLP)** techniques, while emphasizing **ethical AI practices** and **optimization strategies**.  

The work was completed collaboratively by:  
- Wycliffe Arombo  
- Peter Ater Chan  
- Terry Nyambura Mugure  

---

##  Setup & Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/mastering-ai-toolkit.git
cd mastering-ai-toolkit
```

### 2️⃣ Install Dependencies  
```bash
pip install tensorflow numpy matplotlib pandas scikit-learn spacy textblob
python -m spacy download en_core_web_sm
```

**For Google Colab:**  
```python
!pip install spacy textblob tensorflow scikit-learn matplotlib
!python -m spacy download en_core_web_sm
```

---

## Practical Implementation  

### 🔹 Task 1: Classical Machine Learning (Scikit-learn)  
**Dataset:** Iris Species Dataset  
**Goal:** Build and evaluate a Decision Tree Classifier to predict iris flower species.  

**Key Steps:**  
- Handle missing values and encode labels  
- Train/test split and model training  
- Evaluate using accuracy, precision, recall  

**Performance:**  
✅ Accuracy: 100%  
✅ Precision: 100%  
✅ Recall: 100%  

**Tools:** `pandas`, `scikit-learn`

---

### 🔹 Task 2: Deep Learning (TensorFlow)  
**Dataset:** MNIST Handwritten Digits  
**Goal:** Train a Convolutional Neural Network (CNN) to classify handwritten digits.  

**Achievements:**  
- Model achieved **>95% accuracy**  
- Visualized predictions for sample images  
- Integrated **TensorBoard** for model tracking  

**Commands:**  
```bash
python AI_wk3.py
tensorboard --logdir logs/fit
```
Open your browser at [http://localhost:6007](http://localhost:6007)

**Tools:** `tensorflow`, `keras`, `matplotlib`

---

### 🔹 Task 3: Natural Language Processing (spaCy)  
**Dataset:** Sample Amazon Product Reviews  
**Goal:** Perform NER and Sentiment Analysis  

**Approach:**  
- Named Entity Recognition using `spaCy`  
- Sentiment polarity with `TextBlob`  

**Example Output:**  
```
Review: I love the new Apple iPhone!
Entities: Apple iPhone (ORG)
Sentiment: Positive
```

**Tools:** `spaCy`, `TextBlob`

---

## Results Summary  

| Task | Framework | Dataset | Key Metric | Result |
|------|------------|----------|-------------|---------|
| Task 1 | Scikit-learn | Iris Dataset | Accuracy | ~100% |
| Task 2 | TensorFlow | MNIST | Test Accuracy | >95% |
| Task 3 | spaCy + TextBlob | Amazon Reviews | Sentiment | Positive/Negative |

**Visual outputs** (graphs, TensorBoard logs, and screenshots) are located in the `logs/fit/` and `screenshots/` directories.

---

##  Files Included  

| File | Description |
|------|--------------|
| `AI_wk3.py` | Main script integrating ML, DL, and NLP tasks |
| `trained-model.py` | Script for saving/reloading the trained CNN model |
| `Part1_Theoretical_Understanding.pdf` | Theory component of the project |
| `Screenshots_Section_Introduction.pdf` | Screenshot section proof |
| `AI_wk3_screenshot.pdf` | Final execution results |
| `README.md` | Documentation file |

---

##  Ethical Considerations  

**Identified Biases:**  
- MNIST handwriting diversity may limit fairness.  
- Amazon review sentiment may reflect polarity bias.  

**Mitigation Steps:**  
- Applied **TensorFlow Fairness Indicators** for fairness metrics.  
- Balanced sentiment weighting using spaCy filters.  
- Used diverse and representative samples for text analysis.  

---

## References  

- [TensorFlow Documentation](https://www.tensorflow.org)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable)  
- [spaCy Documentation](https://spacy.io)  
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)  

---

##  Git Workflow  

Push all files (code, PDFs, screenshots, README) to GitHub:  

```bash
git add .
git commit -m "Final AI Toolkit project with all code, screenshots, and documentation"
git push origin main
```

---

## 🏷️ Repository Tags  
`AI` · `Machine Learning` · `Deep Learning` · `TensorFlow` · `Scikit-learn` · `spaCy` · `NLP` · `Python` · `Data Science` · `PLP Academy`  

---

**✅ Project Completed Successfully — All tasks executed, logged, and documented for submission.**
````

---

