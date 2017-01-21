from __future__ import division
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from pyearth import Earth
from matplotlib import pyplot

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
'''
import ggplot 
from ggplot import *
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression


from anfis import anfis
from anfis import membership #import membershipfunction, mfDerivs

'''
See paper below

Model comparison is done using Regression Error Characteristics Curves as described in Jinbo Bi, Kristin Bennett. Regression error characteristic curves. In Proceedings of the 20th international conference on machine learning, pages 43--50, 2003.
'''

df = pd.read_csv('data/data.csv')


predictorLabel = df.columns[:-8].tolist()

#['Lean Amine Pressure (bar)', 'Lean Amine Circulation Rate (t/d)', 'Lean Amine Temperature (C)', 'Heat Duty (GJ/h)', 'Reboiler Pressure (bar (g))', 'MDEA Concentration (%)', 'PZ Concentration (%)']



#['LP', 'LC', 'LT', 'HD', 'RP', 'MC', 'PZ']

print "List of Attributes"
print predictorLabel

def mean(a):
    return sum(a) / len(a)


def max_value(inputlist):
    '''
        maximum value from list of lists
    '''
    return max([sublist[-1] for sublist in inputlist])



def getPredictorData (df):
    '''
        get the predictor variable
    '''

    X = df[predictorLabel ].values
    return X


def getLabelData (df, label):
    '''
        get the predicted variable
    '''
    outputLabel = [label]
    y = df[outputLabel ].values

    return y


def evaluate (model, X, y):
    '''
        get the coefficient of determination
    '''
    scoresr2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("R^2: %0.4f" % scoresr2.mean() )
    return scoresr2.mean()




def getHeightOfDecisionTreeRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow):
    "return optimal height"

    rsquaredList = [0]*6
    heighList = [0]*6
    for maxind in range (5, 100, 5):
        clf = DecisionTreeRegressor(max_depth=maxind, random_state=10)

        r2 = evaluate (clf, X, y_sweetgasco2)
        if rsquaredList[0] < r2:
            heighList[0] = maxind
            rsquaredList[0] = r2

        r2 = evaluate (clf, X, y_sweetgasc1)
        if rsquaredList[1] < r2:
            heighList[1] = maxind
            rsquaredList[1] = r2

        r2 = evaluate (clf, X, y_richaminehydro)
        if rsquaredList[2] < r2:
            heighList[2] = maxind
            rsquaredList[2] = r2

        r2 = evaluate (clf, X, y_richaminehco3)
        if rsquaredList[3] < r2:
            heighList[3] = maxind
            rsquaredList[3] = r2

        r2 = evaluate (clf, X, y_sweetgasmdeaflow)
        if rsquaredList[4] < r2:
            heighList[4] = maxind
            rsquaredList[4] = r2

        r2 = evaluate (clf, X, y_sweetgaspzflow)
        if rsquaredList[5] < r2:
            heighList[5] = maxind
            rsquaredList[5] = r2

    return     rsquaredList, heighList


def getHeightOfRandomForestRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow):
    "return optimal height"

    rsquaredList = [0]*6
    heighList = [0]*6
    estimatList = [0]*6
    for nestimator in range (30, 500, 30):
        for maxind in range (5, 100, 5):
            clf = RandomForestRegressor(max_depth=maxind, n_estimators=nestimator, random_state=10)

            r2 = evaluate (clf, X, y_sweetgasco2)
            if rsquaredList[0] < r2:
                heighList[0] = maxind
                estimatList[0] = nestimator
                rsquaredList[0] = r2

            r2 = evaluate (clf, X, y_sweetgasc1)
            if rsquaredList[1] < r2:
                heighList[1] = maxind
                estimatList[1] = nestimator
                rsquaredList[1] = r2

            r2 = evaluate (clf, X, y_richaminehydro)
            if rsquaredList[2] < r2:
                heighList[2] = maxind
                estimatList[2] = nestimator
                rsquaredList[2] = r2

            r2 = evaluate (clf, X, y_richaminehco3)
            if rsquaredList[3] < r2:
                heighList[3] = maxind
                estimatList[3] = nestimator
                rsquaredList[3] = r2

            r2 = evaluate (clf, X, y_sweetgasmdeaflow)
            if rsquaredList[4] < r2:
                heighList[4] = maxind
                estimatList[4] = nestimator
                rsquaredList[4] = r2

            r2 = evaluate (clf, X, y_sweetgaspzflow)
            if rsquaredList[5] < r2:
                heighList[5] = maxind
                estimatList[5] = nestimator
                rsquaredList[5] = r2

    return     rsquaredList, heighList, estimatList 


def getHeightOfExtraTreesRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow):
    "return optimal height"

    rsquaredList = [0]*6
    heighList = [0]*6
    estimatList = [0]*6
    for nestimator in range (30, 500, 30):
        for maxind in range (5, 100, 5):
            clf = ExtraTreesRegressor(max_depth=maxind, n_estimators=nestimator, random_state=10)

            r2 = evaluate (clf, X, y_sweetgasco2)
            if rsquaredList[0] < r2:
                heighList[0] = maxind
                estimatList[0] = nestimator
                rsquaredList[0] = r2

            r2 = evaluate (clf, X, y_sweetgasc1)
            if rsquaredList[1] < r2:
                heighList[1] = maxind
                estimatList[1] = nestimator
                rsquaredList[1] = r2

            r2 = evaluate (clf, X, y_richaminehydro)
            if rsquaredList[2] < r2:
                heighList[2] = maxind
                estimatList[2] = nestimator
                rsquaredList[2] = r2

            r2 = evaluate (clf, X, y_richaminehco3)
            if rsquaredList[3] < r2:
                heighList[3] = maxind
                estimatList[3] = nestimator
                rsquaredList[3] = r2

            r2 = evaluate (clf, X, y_sweetgasmdeaflow)
            if rsquaredList[4] < r2:
                heighList[4] = maxind
                estimatList[4] = nestimator
                rsquaredList[4] = r2

            r2 = evaluate (clf, X, y_sweetgaspzflow)
            if rsquaredList[5] < r2:
                heighList[5] = maxind
                estimatList[5] = nestimator
                rsquaredList[5] = r2

    return     rsquaredList, heighList, estimatList 


if __name__ == "__main__":

    X = getPredictorData (df)

    y_sweetgasco2 = getLabelData (df, 'Sweet Gas CO2 (ppm)')
    y_sweetgasc1 = getLabelData (df, 'Sweet Gas C1 (ppm)')
    y_richaminehydro = getLabelData (df, 'Rich Amine Hydrocarbons (t/d)')

    y_richaminehco3 = getLabelData (df, 'R Amine HCO3 (mol/L)')
    y_sweetgasmdeaflow = getLabelData (df, 'Sweet Gas MDEA Flow (t/d)')
    y_sweetgaspzflow = getLabelData (df, 'Sweet Gas PZ Flow (t/d)')


    y_sweetgasco2 = y_sweetgasco2.ravel() 
    y_sweetgasc1 = y_sweetgasc1.ravel() 
    y_richaminehydro = y_richaminehydro.ravel() 

    y_richaminehco3 = y_richaminehco3.ravel() 
    y_sweetgasmdeaflow = y_sweetgasmdeaflow.ravel() 
    y_sweetgaspzflow = y_sweetgaspzflow.ravel()

    mdeaCol = X [:,[4]]
    pzCol = X [:,[5]]

    mdea_pzratio = mdeaCol /  pzCol


    X = np.hstack((  X, mdea_pzratio ))

    print "Decision Tree"
    print getHeightOfDecisionTreeRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow)

    print "Random Forest Tree"
    print getHeightOfRandomForestRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow)

    print "Extra Tree"
    print getHeightOfExtraTreesRegressor(X, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow)














