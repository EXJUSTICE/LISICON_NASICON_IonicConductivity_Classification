# -*- coding: utf-8 -*-
"""
Systematic Evaluation of features using Cross Validation

Add in the feature combinations
Consider moving datapoints to evaluate to test set to reduce test accuracy

After each fitting, we should append it onto a one single csv file only.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it

#https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--inputdir', help='Input directory for txt', type=str, required=True)
#parser.add_argument('--outputdir', help='Output directory for csv', type=str, required=True)


args = parser.parse_args()

base_dir = getattr(args, 'inputdir')
# python script.py --inputdir /path/where/to/work
#out_dir = getattr(args, 'outputdir')

data = pd.read_csv(base_dir+'reviewdatainput1302newfeatures.txt', delimiter = "\t")


print("chk point - 1")


#Basic inspection
#print(data)
print(data.columns)
#print(data.describe())

#Select features vs y - Column of 1 or 0
y = data.Sclass


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

featuresall = ['a' ,'c','a/c','h','d','Vcalc','D1vol','D3vol','D1ionicr','D2ionicr','D1eff','D2eff','D3eff', 'Ga', 'Sc', 'Y', 'Yb', 'Cr', 'Al',
       'In', 'Co', 'Cu', 'Zn', 'Mg', 'Mo', 'Nb', 'Sn', 'Hf', 'Ge', 'Ti', 'Zr',
       'SiO4', 'Na', 'PO4','Fe', 'D1eneg','D2eneg','D3eneg','D1eneff','D2eneff','D3eneff','d2occu']


featuresfilter = ['a' ,'a/c','h','d','D1vol','D3vol','D1ionicr','D2ionicr','D3ionicr','D1eff','D2eff','D3eff', 'Ga', 'Sc', 'Y', 'Yb', 'Cr', 'Al',
       'In', 'Co', 'Cu', 'Zn', 'Mg', 'Mo', 'Nb', 'Sn', 'Ti', 'Zr',
       'SiO4', 'Na', 'PO4','Fe', 'D1eneg','D2eneg','D3eneg','D1eneff','d2occu']

featuresreducednoionic = ['Ea','a' ,'c','Vcalc', 'Ga', 'Sc', 'Y', 'Yb', 'Cr', 'Al',
       'In', 'Co', 'Cu', 'Zn', 'Mg', 'Mo', 'Nb', 'Sn', 'Hf', 'Ge', 'Ti', 'Zr',
       'SiO4', 'Na', 'PO4','Fe']
featuresselected = ['Ea' ,'d2occu','a','c','Vcalc','D1vol','D2ionicr','D1eff','D2eff','D2vol','D3vol', 'Na', 'PO4','Cr','SiO4','Zr','Ti','In','D3eff','Sc','Al','Hf'
]
featuresselectednoelements = ['Ea','a' ,'c','Vcalc','D1vol','D2ionicr','D1eff','D2eff','D2vol','D3vol', 'Na', 'PO4','SiO4','D3eff']

X = data[featuresselected]



#Cross validation method
#In order to evaluate all of the coefficients, we must keep the logistic regression classifier arguments CONSTANT
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler




#Placeholders for features and accuracies
featureholder=[]
scoreholder = []
std= []

combs = it.combinations(featuresfilter,8)

for featurecomb in combs:
    
    X = data[list(featurecomb)]

    sc_X = StandardScaler()
    df = sc_X.fit_transform(X)
    
    clf =RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=5, max_features=4,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    resultstring =str(scores.mean())
    if(scores.mean()>0.87):
        featureholder.append(str(featurecomb))
        scoreholder.append(str(resultstring))
        std.append(str(scores.std()*2))
    
    #Dont print if theres too many, becomes really slow
    #print(featureholder)
    #print(scoreholder)


df = pd.DataFrame()

df['Accuracy'] = scoreholder
df['Features']= featureholder
df['std'] = std

df.to_csv('output_rf.txt', sep='\t', index=False)

"""
Simple Training Script without Cross Validation
Extract coefficients from a fit
best_model = clf.fit(df, y)

#Training accuracy
print(best_model.score)
"""
"""
Predictions

test_dir = 'C:/Users/Adrian/Desktop/MSML/Matminer/TestSet/testsetv2csv.csv'
testdata = pd.read_csv(test_dir)
Xtest = testdata[featuresselected]
testreadyX =  sc_X.fit_transform(Xtest)
Ytest = testdata.Sclass

#TestAccuracy
print(clf.score(Xtest,Ytest))

#View individual predictions
predictions = clf.predict(testreadyX)
for i in range(0, INSERTRANGE):
        print( "Predicted outcome " +str(predictions[i]))
"""