import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

## Load Data
data = pd.read_csv("titanic_data.csv")

## Data Preprocessing
#### Create Numerical representation of 'Sex'
data['SexNum'] = data['Sex'].apply(lambda x: 0 if x=='female' else 1)

#### Fill in missing values in 'Age' with val of 30
#### val of 30 is an approximate mean found in training set exploration
data['Age'].fillna(30, inplace = True)

## Feature Selection
features = ['Pclass', 'SexNum', 'SibSp', 'Parch', 'Age'] # 'Fare'
label = 'Survived'

features_data = data[features]
labels_data = data[label]

## Train, Test, Cross Validation Data Split
## train : test : cv = 0.6 : 0.2 : 0.2
test_num = int(len(features_data)*.2)
features_train_test, features_cv, labels_train_test, labels_cv = train_test_split(features_data,\
                                                                 labels_data, test_size=test_num,\
                                                                 random_state = 94)
features_train, features_test, labels_train, labels_test = train_test_split(features_train_test, \
                                                           labels_train_test, test_size=test_num, \
                                                           random_state = 87)

## Decision Tree Classifier
def DTC(train_X, train_y, test_X, test_y, criterions, min_samples_splits):
    '''examine the best parameter setting'''
    acc = np.zeros((len(criterions), len(min_samples_splits)), dtype=float)
    max_acc = 0
    max_crt = ''
    max_spl = 0
    max_clf = None
    for rid, c in enumerate(criterions):
        for cid, mss in enumerate(min_samples_splits):
            clf = DecisionTreeClassifier(criterion=c, min_samples_split=mss)
            clf.fit(train_X, train_y)
            acc[rid, cid] = clf.score(test_X, test_y)
            if acc[rid, cid] > max_acc:
                max_acc = acc[rid, cid]
                max_crt = criterions[rid]
                max_spl = min_samples_splits[cid]
                max_clf = clf
    return acc, (max_acc, max_crt, max_spl, max_clf)

crts = ['gini', 'entropy']
splits = [10, 20, 30]
acc, best_param = DTC(features_train, labels_train, features_test, labels_test, crts, splits)
print "best accuracy: ", best_param[0]
print "criterion setting which achieves the best accuracy: ", best_param[1]
print "min_samples_split which achieves the best accuracy: ", best_param[2]

#### Model performance in cross validation set
best_clf = best_param[-1]
acc_cv = best_clf.score(features_cv, labels_cv)
print "model surivial outcome prediction accuracy performance in cross validation set: ", acc_cv