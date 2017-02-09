from __future__ import division
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from pyearth import Earth

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

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


def evalEnsembleModel (classifierlist, X, y, n_folds=5):
    '''
        get the R2 of an ensemble on cross validation
    '''
    xOutput = []
    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)

    for train, test in skf.split(sX, sy) :
        X_train = [X[index] for index in list (train)]
        X_test = [X[index] for index in list (test)]
        y_train = [y[index] for index in list (train)]
        y_test = [y[index] for index in list (test)]


        output = np.empty([ len(X_test), len(classifierlist) ])

        for i, cls in enumerate ( classifierlist ): 
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            output[:, i] =  y_pred

        avgOutput = np.mean(output, axis=1)

        r2Val = r2_score(y_test, avgOutput)
        xOutput.append (r2Val)

    result = sum (xOutput) / len (xOutput) 

    print("R^2: %0.4f" % result )
    return result



def crossValScore (cls, X, y, n_folds=5):
    '''
        get the R2 of an ensemble on cross validation
    '''
    xOutput = []
    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)

    for train, test in skf.split(sX, sy) :
        X_train = np.array([X[index] for index in list (train)])
        X_test = np.array([X[index] for index in list (test)])
        y_train = np.array([y[index] for index in list (train)])
        y_test = np.array([y[index] for index in list (test)])

        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        y_pred = np.array (y_pred.T.tolist()[0] )

        r2Val = r2_score( y_test, y_pred )
        xOutput.append (r2Val)

    result = sum (xOutput) / len (xOutput) 

    print("R^2: %0.4f" % result )
    return result



def featureImportanceCV(model, df, y):
    output = []
    for var in predictorLabel:
        labelSet = list ( set(predictorLabel) - set([var]) )
        X = np.array (df[labelSet])
        #r2Val =  evaluate (model, X, y)
        r2Val = crossValScore (model, X, y)


        output.append( r2Val )

    m = max(output)
    maxLst = [predictorLabel[i] for i, j in enumerate(output) if j == m]

    m = min(output)
    minLst = [predictorLabel[i] for i, j in enumerate(output) if j == m]


    sortedElem = [ predictorLabel[i[0]] for i in sorted(enumerate(output), key=lambda x:x[1])]
    print "Attributes in order of importance"
    print sortedElem

    print "Least important feature is " +  ', '.join(maxLst)
    print "Most important feature is " +   ', '.join(minLst)
    return output




def crossValREC (cls,  X, y, n_folds=5 ):
    '''
        get the R2 of an ensemble on cross validation
    '''
    output = {}
    xlist = []
    ylist = []

    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)

    aucList = []
    fprList, tprList = [], []

    for train, test in skf.split(sX, sy) :
        X_train = np.array([X[index] for index in list (train)])
        X_test = np.array([X[index] for index in list (test)])
        y_train = np.array([y[index] for index in list (train)])
        y_test = np.array([ y[index] for index in list (test)])

        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        xL, yL =  genRECData (y_pred, y_test)



        xlist.append ( xL )
        ylist.append ( yL )


    output["x"] = xlist
    output["y"] = ylist
    return output



def crossValRECCV (cls,  X, y, n_folds=5 ):
    '''
        get the R2 of an ensemble on cross validation
    '''
    output = {}
    xlist = []
    ylist = []

    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)

    aucList = []
    fprList, tprList = [], []

    for train, test in skf.split(sX, sy) :
        X_train = np.array([X[index] for index in list (train)])
        X_test = np.array([X[index] for index in list (test)])
        y_train = np.array([y[index] for index in list (train)])
        y_test = np.array([ y[index] for index in list (test)])

        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        y_pred = np.array (y_pred.T.tolist()[0] )

        xL, yL =  genRECData (y_pred, y_test)



        xlist.append ( xL )
        ylist.append ( yL )


    output["x"] = xlist
    output["y"] = ylist
    return output


def crossValNULLREC ( X, y, n_folds=5 ):
    output = {}
    xlist = []
    ylist = []

    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)

    aucList = []
    fprList, tprList = [], []

    for train, test in skf.split(sX, sy) :
        X_train = np.array([X[index] for index in list (train)])
        X_test = np.array([X[index] for index in list (test)])
        y_train = np.array([y[index] for index in list (train)])
        y_test = np.array([ y[index] for index in list (test)])

        avg = np.average(y_train)
        y_pred = avg * np.ones( len( X_test ) )

        xL, yL =  genRECData (y_pred, y_test) 

        xlist.append ( xL )
        ylist.append ( yL )


    output["x"] = xlist
    output["y"] = ylist
    return output




def genRECData (y_pred, y_test):
    abDev = np.fabs(y_pred - y_test) #absolute deviation
    xlist = []
    ylist = []

    abDev = sorted (abDev)
    eprev , correct = 0, 0
    for eVal in abDev[:-1]:
        if eVal > eprev:
            xlist.append ( eVal )
            ylist.append ( correct / (len (abDev)-1) )
            eprev = eVal
        correct = correct + 1

    xlist.append ( abDev[-1] )
    ylist.append ( correct /  (len (abDev)-1) )

    return xlist, ylist 


def AUC( x, y):
    '''
        Area under the curve
    '''
    return np.trapz(y=x, x=y)


def featureImportance(model, df, y):
    output = []
    for var in predictorLabel:
        labelSet = list ( set(predictorLabel) - set([var]) )
        X = df[labelSet]
        #r2Val =  evaluate (model, X, y)
        scoresr2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2Val = scoresr2.mean() 

        output.append( r2Val )

    m = max(output)
    maxLst = [predictorLabel[i] for i, j in enumerate(output) if j == m]

    m = min(output)
    minLst = [predictorLabel[i] for i, j in enumerate(output) if j == m]


    sortedElem = [ predictorLabel[i[0]] for i in sorted(enumerate(output), key=lambda x:x[1])]
    print "Attributes in order of importance"
    print sortedElem

    print "Least important feature is " +  ', '.join(maxLst)
    print "Most important feature is " +   ', '.join(minLst)
    return output



def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(7, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def modelValidation (clf, X, y ):
    '''
        
        return list of area of the Cv, average x, y of REC curve
    '''
    dictVal = crossValREC(clf, X, y )
    xlistOfLists =  dictVal["x"]
    ylistOfLists =  dictVal["y"]
    areaList = []
    for xlist, ylist in zip (xlistOfLists, ylistOfLists):
        aucVal = AUC( xlist, ylist )
        #Calculate the area over the curve
        xlistSort = sorted (xlist)

        totalAreaOfBox = xlistSort[-1]
        aocVal = totalAreaOfBox - aucVal
        areaList.append ( aocVal )

    xAvgList =  map(mean, zip(*xlistOfLists))
    yAvgList =  map(mean, zip(*ylistOfLists))

    aucValAvg = AUC( xAvgList, yAvgList )
    #Calculate the area over the curve
    xlistSort = sorted (xAvgList)

    totalAreaOfBox = xlistSort[-1]
    aocValAvg = totalAreaOfBox - aucValAvg


    return { "areaList": areaList, "xAvgList": xAvgList, "yAvgList": yAvgList, "avgArea": aocValAvg }




def modelValidationCV (clf, X, y ):
    '''
        
        return list of area of the Cv, average x, y of REC curve
    '''
    dictVal = crossValRECCV(clf, X, y )
    xlistOfLists =  dictVal["x"]
    ylistOfLists =  dictVal["y"]
    areaList = []
    for xlist, ylist in zip (xlistOfLists, ylistOfLists):
        aucVal = AUC( xlist, ylist )
        #Calculate the area over the curve
        xlistSort = sorted (xlist)

        totalAreaOfBox = xlistSort[-1]
        aocVal = totalAreaOfBox - aucVal
        areaList.append ( aocVal )

    xAvgList =  map(mean, zip(*xlistOfLists))
    yAvgList =  map(mean, zip(*ylistOfLists))

    aucValAvg = AUC( xAvgList, yAvgList )
    #Calculate the area over the curve
    xlistSort = sorted (xAvgList)

    totalAreaOfBox = xlistSort[-1]
    aocValAvg = totalAreaOfBox - aucValAvg


    return { "areaList": areaList, "xAvgList": xAvgList, "yAvgList": yAvgList, "avgArea": aocValAvg }



def modelNULLValidation (X, y ):
    '''
        
        return list of area of the Cv, average x, y of REC curve
    '''
    dictVal = crossValNULLREC ( X, y )
    xlistOfLists =  dictVal["x"]
    ylistOfLists =  dictVal["y"]
    areaList = []
    for xlist, ylist in zip (xlistOfLists, ylistOfLists):
        aucVal = AUC( xlist, ylist )
        areaList.append ( aucVal )

    xAvgList =  map(mean, zip(*xlistOfLists))
    yAvgList =  map(mean, zip(*ylistOfLists))

    aucValAvg = AUC( xAvgList, yAvgList )
    #Calculate the area over the curve
    xlistSort = sorted (xAvgList)

    totalAreaOfBox = xlistSort[-1]
    aocValAvg = totalAreaOfBox - aucValAvg


    return { "areaList": areaList, "xAvgList": xAvgList, "yAvgList": yAvgList, "avgArea": aocValAvg }



def crossValNULLScore ( X, y, n_folds=5 ):
    xOutput = []
    skf = StratifiedKFold(n_splits=n_folds)

    nlen = ( len (X) // n_folds ) * n_folds

    X = X[:nlen]
    y = y[:nlen]

    sX = np.ones( nlen )
    sy = np.ones( nlen )
    skf = StratifiedKFold(n_splits=n_folds)


    for train, test in skf.split(sX, sy) :
        X_train = np.array([X[index] for index in list (train)])
        X_test = np.array([X[index] for index in list (test)])
        y_train = np.array([y[index] for index in list (train)])
        y_test = np.array([ y[index] for index in list (test)])

        avg = np.average(y_train)
        y_pred = avg * np.ones( len( X_test ) )

        r2Val = r2_score(y_test, y_pred)
        xOutput.append (r2Val)

    result = sum (xOutput) / len (xOutput) 

    print("R^2: %0.4f" % result )
    return result


def drawRECCURVE ( recObjectList, rsquaredList, label  ):
    '''
        accepts list of REC Objects, list of rsquared
    '''
    df = pd.DataFrame()
    clsLabel = ["MARS", "Decision Tree", "ANFIS", "Mean"]

    for ind, rec in enumerate(recObjectList):
        dfA = pd.DataFrame()
        dfA['x'] = rec['xAvgList']
        dfA['y'] = rec['yAvgList']
        dfA['group'] = [ind]*len( rec['yAvgList'] )
        area = "%.2f" % round( rec['avgArea'], 2 )
        rsquared = "%.2f" % round( rsquaredList[ind], 2 )
        #legend = clsLabel[ind] + "(AOC=" + area + ",R2=" + rsquared +")"
        legend = clsLabel[ind]
        dfA['Classifier'] = [legend]*len( rec['yAvgList'] )
        df = df.append(dfA, ignore_index=True)


    p = ggplot(df, aes(x='x', y='y', color='Classifier', group='group'))  + theme_bw()



    nullModel = recObjectList[-1]
    xMax = nullModel['xAvgList'][-1]

    #pVal =  p + geom_point() + geom_line() +  scale_y_continuous(limits=(0,1)) +  scale_x_continuous(limits=(0,xMax)) + ggtitle('REC curve comparing the models for label: (' +label +')') + xlab('Absolute deviation') + ylab('Accuracy')  

    pVal =  p + geom_line() +  scale_y_continuous(limits=(0,1)) +  scale_x_continuous(limits=(0,xMax)) + ggtitle('REC curve comparing the models for label: (' +label +')') + xlab('Absolute deviation') + ylab('Accuracy')  + theme_bw()

    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/rec/"+file_name+".png")



def drawFeatureImportance ( featureImpList, label="variableImportance" ):
    '''
        accepts list of REC Objects, list of rsquared
    '''
    df = pd.DataFrame()
    clsLabel = ["MARS", "Decision Tree", "ANFIS"]

    yLabel = ['Sweet Gas CO2 (ppm)', 'Sweet Gas C1 (ppm)', 'Rich Amine Hydrocarbons (t/d)', 'R Amine HCO3 (mol/L)', 'Sweet Gas PZ Flow (t/d)']


    predLabel = ['LP', 'LF', 'LT', 'HD', 'MD', 'PZ']


    for ind, rec in enumerate(featureImpList):
        for innerind, innerrec in enumerate(rec):
            dfA = pd.DataFrame()
            dfA['Attributes'] = predLabel
            dfA['R2'] = innerrec
            legend = clsLabel[ind]
            dfA['Classifier'] = [legend]*len( innerrec )

            dfA['ylabel'] = [yLabel[innerind]]*len( innerrec )

            df = df.append(dfA, ignore_index=True)

    #pVal = ggplot(df, aes(x='Attributes', weight='R2')) + geom_bar(color='teal') + scale_fill_identity() +  facet_wrap('Classifier', 'ylabel') + ggtitle('Estimation of the importance of variable using R2') + xlab('Attributes') + ylab('R2') + theme(axis_text_x  = element_text(angle = 90)) +  scale_x_continuous(breaks=[0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8],  labels=predictorLabel)

    pVal = ggplot(df, aes(x='Attributes', weight='R2')) + geom_bar(color='teal') + scale_fill_identity() +  facet_wrap('Classifier', 'ylabel') + ggtitle('Estimation of the importance of variable using R2') + xlab('Attributes') + ylab('R2')  + theme_bw()

    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/"+file_name+".png")


def drawTree (clf, X, y, filename):
    clf = clf.fit(X, y)
    tree.export_graphviz(clf, out_file="temp/"+filename+'.dot')  
    cmd = "dot -Tpng temp/"+ filename + ".dot -o pictures/tree/" + filename + ".png"
    os.system(cmd)



def drawParityChart ( xdata, ydata, label ):
    '''
        plot parity chart
    '''
    df = pd.DataFrame()
    df['x'] = xdata
    df['y'] = ydata

    #pVal = ggplot(df, aes(x='x', y='y')) + geom_point() + ggtitle('Parity Chart: '+label) + xlab('Experimental Value') + ylab('Predicted Value') + geom_abline()

    r2Val = r2_score(  xdata, ydata )

    pVal = ggplot(df, aes(x='x', y='y')) + geom_point(color='blue') + ggtitle('Parity Chart: '+label+' | ' +"R^2: %0.4f" % r2Val ) + xlab('Experimental Value') + ylab('Predicted Value')  + stat_smooth( se=False )  + theme_bw()

    print("R^2: %0.4f" % r2Val )


    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/parity/"+file_name+".png")



def unison_shuffled_copies(a, b, c, d, e, f, g):
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p], e[p], f[p], g[p]



def unison_shuffled(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


class AnfisClassifier:
    'ANFIS classifier'


    def __init__(self):
        '''
            Constructor
        '''

        self.anfis = anfis.ANFIS
        self.mem = membership.membershipfunction
        self.pred = anfis.predict

        self.type = 'gaussmf'



    def setType (self, type):
        self.type = type


    def parameters (self, X):
        self.X = X



    def fitbell(self, X, y, epochs=10):

        mf = []


        meanList = self.X.mean( axis=0 )
        stdList = self.X.std(axis=0) 

        minArr = np.amin( self.X, axis=0 )
        maxArr  = np.amax( self.X, axis=0 )


        for ind in range(len(meanList)):
            #temp  = [['gbellmf',{'a': round(minArr[ind]),'b': round(meanList[ind]), 'c': round(maxArr[ind]) }]]

            #temp  = [['sigmf',{'b': round(minArr[ind]), 'c': round(meanList[ind]) }]]

            aVal = minArr[ind] + 0.2 * ( maxArr[ind] - minArr[ind] )
            bVal = minArr[ind] + 0.4 * ( maxArr[ind] - minArr[ind] )
            cVal = minArr[ind] + 0.6 * ( maxArr[ind] - minArr[ind] )

            #temp  = [['sigmf',{'b': bVal, 'c': meanList[ind]  }]]
            temp  = [['gbellmf',{'a': aVal,'b': bVal, 'c': cVal }]]
            mf.append (temp)


        #mfc = membership.membershipfunction.MemFuncs(mf)
        mfc = self.mem.MemFuncs(mf)
        #self.anf = anfis.ANFIS(X, y, mfc)
        self.anf = self.anfis(X, y, mfc)
        self.anf.trainHybridJangOffLine(epochs=epochs)



    def fitgauss(self, X, y, epochs=10):

        mf = []

        meanList = X.mean(axis=0)
        stdList = X.std(axis=0) 

        for ind in range(len(meanList)):
            temp  = [['gaussmf',{'mean':meanList[ind],'sigma':stdList[ind]}]]
            mf.append (temp)

        #mfc = membership.membershipfunction.MemFuncs(mf)
        mfc = self.mem.MemFuncs(mf)
        #self.anf = anfis.ANFIS(X, y, mfc)
        self.anf = self.anfis(X, y, mfc)
        self.anf.trainHybridJangOffLine(epochs=epochs)


    def fitsigmf(self, X, y, epochs=10):

        mf = []


        meanList = self.X.mean( axis=0 )
        stdList = self.X.std(axis=0) 

        minArr = np.amin( self.X, axis=0 )
        maxArr  = np.amax( self.X, axis=0 )


        for ind in range(len(meanList)):

            bVal = minArr[ind] + 0.2 * ( maxArr[ind] - minArr[ind] )
            cVal = minArr[ind] + 0.6 * ( maxArr[ind] - minArr[ind] )


            temp  = [['sigmf',{'b': bVal, 'c': cVal }]]
            mf.append (temp)


        #mfc = membership.membershipfunction.MemFuncs(mf)
        mfc = self.mem.MemFuncs(mf)
        #self.anf = anfis.ANFIS(X, y, mfc)
        self.anf = self.anfis(X, y, mfc)
        self.anf.trainHybridJangOffLine(epochs=epochs)




    def fit(self, X, y, epochs=10):

        if self.type == 'gbellmf':
            self.parameters (X )
            self.fitbell( X, y, epochs )


        if self.type == 'gaussmf':
            self.fitgauss( X, y, epochs)


        if self.type == 'sigmf':
            self.parameters (X )
            self.fitsigmf( X, y, epochs)

    def predict(self, X):
        #return anfis.predict( self.anf, X )
        return self.pred( self.anf, X )


'''
    How to use program

    anfis = AnfisClassifier()
    anfis.fit(X, y_sweetgasc1)
    print anfis.predict(X)
'''




if __name__ == "__main__":

    X = getPredictorData (df)

    y_sweetgasco2 = getLabelData (df, 'Sweet Gas CO2 (ppm)')
    y_sweetgasc1 = getLabelData (df, 'Sweet Gas C1 (ppm)')
    y_richaminehydro = getLabelData (df, 'Rich Amine Hydrocarbons (t/d)')

    y_richaminehco3 = getLabelData (df, 'R Amine HCO3 (mol/L)')
    y_sweetgasmdeaflow = getLabelData (df, 'Sweet Gas MDEA Flow (t/d)')
    y_sweetgaspzflow = getLabelData (df, 'Sweet Gas PZ Flow (t/d)')


    y_sweetgasco2Log = np.log ( y_sweetgasco2.ravel() )
    y_sweetgasc1Log = np.log ( y_sweetgasc1.ravel() )
    y_richaminehydroLog = np.log ( y_richaminehydro.ravel() )

    y_richaminehco3Log = np.log ( y_richaminehco3.ravel() )
    y_sweetgasmdeaflowLog = np.log ( y_sweetgasmdeaflow.ravel() )
    y_sweetgaspzflowLog = np.log ( y_sweetgaspzflow.ravel() )



    y_sweetgasco2 = y_sweetgasco2.ravel() 
    y_sweetgasc1 = y_sweetgasc1.ravel() 
    y_richaminehydro = y_richaminehydro.ravel() 

    y_richaminehco3 = y_richaminehco3.ravel() 
    y_sweetgasmdeaflow = y_sweetgasmdeaflow.ravel() 
    y_sweetgaspzflow = y_sweetgaspzflow.ravel() 



    mdeaCol = X [:,[4]]
    pzCol = X [:,[5]]

    mdea_pzratio = mdeaCol /  pzCol


    cX = np.hstack((  X, mdea_pzratio ))


    ntrainingSize = len ( cX ) - 300

    recListSweetGasCO2 = [] 
    recListSweetGasC1 = []
    recListRichAmineHydro  = []
    recListRichAmineHco3 = [] 
    #recListSweetGasMdeaFlow = []
    recListSweetGaspzFlow  = []


    rsquaredSweetGasCO2 = [] 
    rsquaredSweetGasC1 = [] 
    rsquaredRichAmineHydro = [] 
    rsquaredRichAmineHco3 = [] 
    #rsquaredSweetGasMdeaFlow = []
    rsquaredSweetGaspzFlow  = []


    featureImpList = []

    #shuffling the data
    cX, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow  = unison_shuffled_copies(cX, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow )


    meanList = cX.mean(axis=0)
    meanMatrixlist = []
    for i in range(len(cX)):
        mean_line = meanList.reshape(1,len(meanList))
        meanMatrixlist.append(mean_line)

    meanMatrix = np.vstack(meanMatrixlist)

    zero_meanX = cX - meanMatrix

    """
    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas C02 (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , y_sweetgasco2)
    print "AVG gaussmf R^2: " + str (r2)


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas C1 (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , y_sweetgasc1)
    print "AVG gaussmf R^2: " + str (r2)



    print "----------------------------------------------------------"
    print "Evaluate performance of 'Rich amine hydrocarbon (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , y_richaminehydro)
    print "AVG gaussmf R^2: " + str (r2)


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Rich amine hCO3 (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , y_richaminehco3)
    print "AVG gaussmf R^2: " + str (r2)


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet gas mdea flow (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , y_sweetgasmdeaflow)
    print "AVG gaussmf R^2: " + str (r2)



    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet gas pz flow (ppm)'"
    print "----------------------------------------------------------"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX ,  y_sweetgaspzflow)
    print "AVG gaussmf R^2: " + str (r2)

    """


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet gas mdea flow (ppm)'"
    print "----------------------------------------------------------"

    
    model = AnfisClassifier()
    model.setType ('gaussmf')
    r2 = crossValScore (model, zero_meanX , 1000000000000*y_sweetgasmdeaflow)
    print "AVG gaussmf R^2: " + str (r2)

