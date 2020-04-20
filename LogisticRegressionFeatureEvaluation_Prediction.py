# -*- coding: utf-8 -*-
"""
Systematic evaluation of features for logistic regression model using cross-validation
for the application of NASICON/LISICON ionic conductivity classification

By Xu et al. for J Phys Commun.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


base_dir = '.../TrainingData.csv'
data = pd.read_csv(base_dir)


#Basic inspection
#print(data)
print(data.columns)
#print(data.describe())

#Select features vs y
y = data.Sclass

####################Feature Lists#####################################
featuresall = ['a' ,'c','a/c','h','d','Vcalc','D1vol','D3vol','D1ionicr','D2ionicr','D1eff','D2eff','D3eff', 'Ga', 'Sc', 'Y', 'Yb', 'Cr', 'Al',
       'In', 'Co', 'Cu', 'Zn', 'Mg', 'Mo', 'Nb', 'Sn', 'Hf', 'Ge', 'Ti', 'Zr',
       'SiO4', 'Na', 'PO4', 'D1eneg','D2eneg','D3eneg','D1eneff','D2eneff','D3eneff','d2occu']


featuresfilter = ['a' ,'c','a/c','h','d','Vcalc','D1vol','D3vol','D2vol','D1ionicr','D2ionicr','D1eff','D2eff','D3eff', 'Na', 'PO4', 'D1eneg','D2eneg','D3eneg','D1eneff','D2eneff','D3eneff','d2occu']

featuresreducednoionic = ['Ea','a' ,'c','Vcalc', 'Ga', 'Sc', 'Y', 'Yb', 'Cr', 'Al',
       'In', 'Co', 'Cu', 'Zn', 'Mg', 'Mo', 'Nb', 'Sn', 'Hf', 'Ge', 'Ti', 'Zr',
       'SiO4', 'Na', 'PO4','Fe']
featuresselected = ['Ea' ,'d2occu','a','c','Vcalc','D1vol','D2ionicr','D1eff','D2eff','D2vol','D3vol', 'Na', 'PO4','Cr','SiO4','Zr','Ti','In','D3eff','Sc','Al','Hf'
]
featuresselectednoelements = ['Ea','a' ,'c','Vcalc','D1vol','D2ionicr','D1eff','D2eff','D2vol','D3vol', 'Na', 'PO4','SiO4','D3eff']

featurestest = ['a', 'c', 'h', 'd', 'D1ionicr', 'D2ionicr', 'PO4', 'D1eneg', 'D2eneg', 'D3eneg']


"""
####################Feature Combination Evaluation#####################################

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler




#Placeholders for features and accuracies

featureholder=[]
scoreholder = []
std= []

combs = it.combinations(featuresfilter,5)

for featurecomb in combs:
    
    X = data[list(featurecomb)]
    
    clf =LogisticRegression(C=1.0005691831793622, class_weight=None, dual=False,multi_class='ovr', n_jobs=1, penalty='l2', random_state=2, solver='sag', tol=0.0001, verbose=0, warm_start=False)
    clf.fit(X,y)
    #Training Accuracy
    scores = clf.score(X,y)
    resultstring =str(scores)
    if(scores.mean()>0.8):
        featureholder.append(str(featurecomb))
        scoreholder.append(str(resultstring))
        std.append(str(scores.std()*2))
    
    #Do not invoke print if feature count is high, will slow down repository considerably
    #print(featureholder)
    #print(scoreholder)


df = pd.DataFrame()

df['Accuracy'] = scoreholder
df['Features']= featureholder
df['std'] = std


df.to_csv('Feature_Evaluation_CV.csv')
"""
"""
####################Coefficient Extraction#####################################
Extract coefficients from a fit

best_model = clf.fit(X y)
coef = clf.coef_[0]
print (coef)
#Training accuracy
print(best_model.score)
"""
"""
####################FeedForward Prediction#####################################
from sklearn.preprocessing import StandardScaler


clf =LogisticRegression(C=1.0005691831793622, max_iter=80,class_weight=None, dual=False,multi_class='ovr', n_jobs=1, penalty='l2', random_state=4, solver='sag', tol=0.0001, verbose=0, warm_start=False)

X = data[featurestest]


test_dir = '.../TestSet_NaLi.csv'
testdata = pd.read_csv(test_dir)
Xtest = testdata[featurestest]
Ytest = testdata.Sclass

#TestAccuracy
print(clf.score(Xtest,Ytest))

predictions = clf.predict(Xtest)
prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction_NaLi_testset.csv')
"""