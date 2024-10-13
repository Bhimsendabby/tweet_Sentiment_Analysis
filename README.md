# Tweet Sentiment Analysis using SVM and Word2Vec (CBOW)

This project focuses on tweet sentiment analysis using a dataset of over 400,000 tweets categorized as Positive, Negative, or Neutral. The model is trained using a Support Vector Machine (SVM) and vectorized using Word2Vec CBOW embeddings.

## Table of Contents
1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
   - [URL Removal](#url-removal)
   - [Slang Removal](#slang-removal)
   - [Emoji Removal](#emoji-removal)
   - [Punctuation Removal](#punctuation-removal)
   - [Stop Word Removal](#stop-word-removal)
   - [Lemmatization](#lemmatization)
3. [Tokenization](#tokenization)
4. [Word2Vec CBOW Vectorization](#word2vec-cbow-vectorization)
5. [Handling Null Values](#handling-null-values)
6. [Train-Test Split](#train-test-split)
7. [Label Encoding](#label-encoding)
8. [Model Training](#model-training)
9. [Evaluation Metrics](#evaluation-metrics)
   - [AUC Score and ROC Curve](#auc-score-and-roc-curve)
   - [Train and Test Accuracy](#train-and-test-accuracy)
   - [F1 Score](#f1-score)
   - [Confusion Matrix](#confusion-matrix)
10. [Model Tuning](#model-tuning)
11. [Cross Validation](#cross-validation)

---

## 1. Dataset
The dataset contains over 400,000 tweets with sentiment labels categorized as Positive, Negative, or Neutral.

---

## 2. Preprocessing
The preprocessing steps ensure the dataset is clean and ready for analysis.

### 2.1 URL Removal
All URLs in the tweets are removed using regular expressions to eliminate noise.

### 2.2 Slang Removal
Common internet slangs are replaced with their proper equivalents to standardize the language.

### 2.3 Emoji Removal
All emojis are removed, focusing purely on the textual content for analysis.

### 2.4 Punctuation Removal
Punctuations are removed to clean up the text.

### 2.5 Stop Word Removal
Stop words like "and", "the", and "in" are removed to reduce unnecessary noise in the dataset.

### 2.6 Lemmatization
Lemmatization is applied to reduce words to their base forms (e.g., "running" becomes "run").

---

## 3. Tokenization
After preprocessing, the tweets are tokenized into individual words for analysis.

---

## 4. Word2Vec CBOW Vectorization
Using a Word2Vec CBOW (Continuous Bag of Words) model, the tokenized words are transformed into vector representations. Each token is represented by a 100-dimensional vector.

---

## 5. Handling Null Values
All null values in the dataset are handled by either filling them with appropriate values or removing rows containing them.

---

## 6. Train-Test Split
The dataset is split into training and testing sets to evaluate the model’s performance.

---

## 7. Label Encoding
The sentiment labels (Positive, Negative, Neutral) are encoded into numerical values so that they can be used in the model.

---

## 8. Model Training
The SVM model is trained using the vectorized tweet data, with an emphasis on classification accuracy for the three sentiment classes.

---

## 9. Evaluation Metrics
### 9.1 AUC Score and ROC Curve
The Area Under the Curve (AUC) score and Receiver Operating Characteristic (ROC) curve are calculated to evaluate the model's performance in distinguishing between the sentiment classes.

### 9.2 Train and Test Accuracy
The model's accuracy on both the training and testing sets is calculated to determine its generalization performance.

### 9.3 F1 Score
The F1 score is calculated to balance the trade-off between precision and recall for the multi-class classification task.

### 9.4 Confusion Matrix
A confusion matrix is generated to visually represent the model's performance and classification accuracy for each sentiment class.

---

## 10. Model Tuning
Hyperparameter tuning is performed to optimize the model’s performance and improve its classification capabilities.

---

## 11. Cross Validation
K-Fold cross-validation is applied to validate the model's robustness and ensure that the results are not dependent on a specific train-test split.

---

## Conclusion
This project showcases an end-to-end process of tweet sentiment analysis, from data preprocessing to model evaluation and tuning. The use of Word2Vec embeddings and SVM for classification offers robust performance in detecting sentiments from tweets.

Feel free to explore the code in detail and adapt it to your own projects!
