# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 02:21:51 2017

@author: sys
"""

# import all dependencies
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

# read the data and replace null values with a null string
df1 = pd.read_csv("spamham.csv")
df = df1.where((pd.notnull(df1)), '')

# Categorize Spam as 0 and Not spam as 1 
df.loc[df["Category"] == 'ham', "Category",] = 1
df.loc[df["Category"] == 'spam', "Category",] = 0
# split data as label and text . System should be capable of predicting the label based on the  text
df_x = df['Message']
df_y = df['Category']
# split the table - 80 percent for training and 20 percent for test size
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, test_size=0.2, random_state=4)

# feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_trainFeat = tfvec.fit_transform(x_train)
x_testFeat = tfvec.transform(x_test)

# SVM is used to model
y_trainSvm = y_train.astype('int')
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_trainSvm)
predResult = classifierModel.predict(x_testFeat)

# GNB is used to model
y_trainGnb = y_train.astype('int')
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_trainGnb)
predResult2 = classifierModel2.predict(x_testFeat)

# Calc accuracy,converting to int - solves - cant handle mix of unknown and binary
y_test = y_test.astype('int')
actual_Y = y_test.as_matrix()

print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")
#Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult)*100))
#FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro')*100))
cmSVM=confusion_matrix(actual_Y, predResult)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using SVM:")
print(cmSVM)
print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
#Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResult2)*100))
#FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResult2, average='macro')*100))
cmMNb=confusion_matrix(actual_Y, predResult2)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using MNB:")
print(cmMNb)