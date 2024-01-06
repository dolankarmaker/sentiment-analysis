# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:50:56 2019

@author: HP
"""
# Importing the libraries
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset
# Change the path according to dataset location
dataset = pd.read_csv('D:\\Sentiment Analysis Based on Restaurant Review\\Restaurant_Reviews.tsv', delimiter='\t',
                      quoting=3)

# print(dataset.head)

corpus = []
for i in range(0, 1000):
    # The syntax for re.sub() is re.sub(pattern,repl,string).
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multinomial NB
print("Multinomial NB")
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)
print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# Bernoulli NB
print("Bernoulli NB")
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB(alpha=0.8)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)
print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# Logistic Regression
print("Logistic Regression")
# Fitting Logistic Regression to the Training set
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)
print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# Decision Tree
print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)

print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# SVM
print("SVM")
from sklearn.svm import SVC

classifier = SVC(C=1, kernel='linear', degree=3, gamma='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)

print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# K nearest
print("K nearest")
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=25)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)
print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
print("\n")
print("\n")

# MLP Deep learning
print("MLP Deep Learning")
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred)
score3 = recall_score(y_test, y_pred)

print("Accuracy is ", round(score1 * 100, 2), "%")
print("Precision is ", round(score2, 2))
print("Recall is ", round(score3, 2))
