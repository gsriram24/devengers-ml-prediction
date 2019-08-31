#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:33:33 2019

@author: sgcuber24
"""

from sklearn.svm import SVC
import csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from time import time
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np

dataTrain = pd.read_csv("trainms.csv")
dataTest = pd.read_csv("testms.csv")
dataTest["label"] = "test"
treatment_raw = dataTrain['treatment']

features_raw = dataTrain.drop('treatment', axis=1)
features_raw["label"] = "train"
featuresConcat = pd.concat([features_raw, dataTest], sort=False)
dummy_cols = list(set(featuresConcat.columns) - set(['label']))
concatDummies = pd.get_dummies(featuresConcat, columns=dummy_cols)

features_final = concatDummies[concatDummies['label'] == 'train']
dataTest_final = concatDummies[concatDummies['label'] == 'test']

features_final = features_final.drop('label', axis=1)
dataTest_final = dataTest_final.drop('label', axis=1)
features_final.head()


treatment_final = pd.Series(treatment_raw.apply(lambda x: 1 if x == 'Yes' else 0))


X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    treatment_final,
                                                    test_size=0.2,
                                                    )
X_train, y_train = features_final, treatment_final

X_test = dataTest_final
# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}
    beta = 0.5
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()

    results['train_time'] = end-start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

#
#    results['pred_time'] = end-start
#
#    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
#
#    results['acc_test'] = accuracy_score(y_test, predictions_test)
#
#    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=beta)
#
#    results['f_test'] = fbeta_score(y_test, predictions_test, beta=beta)

#    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
#    print('Acc Test:',results['acc_test'])
    return predictions_test


clf_B = GradientBoostingClassifier()
clf_C = AdaBoostClassifier()

samples_100 = len(y_train)
samples_10 = int(.10 * samples_100)
samples_1 = int(0.01 * samples_100)
for clf in [clf_B]:
    clf_name = clf.__class__.__name__
    for i, samples in enumerate([samples_100]):
        results = train_predict(clf, samples, X_train, y_train, X_test, 0)

a = ["Yes" if i == 1 else "No" for i in results]
b = []
for i in range(len(a)):
    b.append([i+1, a[i]])
print(b)

with open('outputGB.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['s.no', 'treatment'])
    writer.writerows(b)
