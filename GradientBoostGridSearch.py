#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:21:07 2019

@author: sgcuber24
"""
import pandas as pd
import numpy as np

dataTrain = pd.read_csv("trainms.csv")
dataTest = pd.read_csv("testms.csv")
dataTest["label"] = "test"
treatment_raw=dataTrain['treatment']

features_raw = dataTrain.drop('treatment', axis = 1)
features_raw["label"] = "train"
featuresConcat = pd.concat([features_raw,dataTest], sort = False)
dummy_cols = list(set(featuresConcat.columns) - set(['label']))
concatDummies = pd.get_dummies(featuresConcat, columns=dummy_cols)

features_final = concatDummies[concatDummies['label'] == 'train']
dataTest_final = concatDummies[concatDummies['label'] == 'test']

features_final = features_final.drop('label', axis=1)
dataTest_final = dataTest_final.drop('label', axis=1)
features_final.head()


treatment_final = pd.Series(treatment_raw.apply(lambda x: 1 if x == 'Yes' else 0))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    treatment_final, 
                                                    test_size = 0.2, 
                                                    )
X_train,y_train = features_final, treatment_final

X_test = dataTest_final

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, accuracy_score

clf = GradientBoostingClassifier()

parameters = {'n_estimators':[1,10,100], }

scorer = make_scorer(fbeta_score, beta=0.5)

grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring=scorer, cv=5)


grid_fit = grid_obj.fit(X_train, y_train, sample_weight=None)

best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_

from sklearn.base import clone

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:25]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:25]]]


clf = (clone(best_clf)).fit(X_train_reduced, y_train)

reduced_predictions = clf.predict(X_test_reduced)

print(reduced_predictions)

a=["Yes" if i==1 else "No" for i in reduced_predictions]
b=[]
for i in range(len(a)):
    b.append([i+1,a[i]])
print(b)

import csv
with open('outputGBR.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['s.no', 'treatment'])
    writer.writerows(b)