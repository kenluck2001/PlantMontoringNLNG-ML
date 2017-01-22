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
    clsLabel = ["MARS", "Tree", "ANFIS", "Mean"]

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


    p = ggplot(df, aes(x='x', y='y', color='Classifier', group='group'))



    nullModel = recObjectList[-1]
    xMax = nullModel['xAvgList'][-1]

    #pVal =  p + geom_point() + geom_line() +  scale_y_continuous(limits=(0,1)) +  scale_x_continuous(limits=(0,xMax)) + ggtitle('REC curve comparing the models for label: (' +label +')') + xlab('Absolute deviation') + ylab('Accuracy')  

    pVal =  p + geom_line() +  scale_y_continuous(limits=(0,1)) +  scale_x_continuous(limits=(0,xMax)) + ggtitle('REC curve comparing the models for label: (' +label +')') + xlab('Absolute deviation') + ylab('Accuracy')

    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/rec/"+file_name+".png")



def drawFeatureImportance ( featureImpList, label="variableImportance" ):
    '''
        accepts list of REC Objects, list of rsquared
    '''
    df = pd.DataFrame()
    clsLabel = ["Tree", "ANFIS"]

    yLabel = ['Sweet Gas CO2 (ppm)', 'Sweet Gas C1 (ppm)', 'Rich Amine Hydrocarbons (t/d)', 'R Amine HCO3 (mol/L)', 'Sweet Gas MDEA Flow (t/d)', 'Sweet Gas PZ Flow (t/d)']


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

    pVal = ggplot(df, aes(x='Attributes', weight='R2')) + geom_bar(color='teal') + scale_fill_identity() +  facet_wrap('Classifier', 'ylabel') + ggtitle('Estimation of the importance of variable using R2') + xlab('Attributes') + ylab('R2')

    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/"+file_name+".png")


def drawTree (clf, X, y, filename):
    clf = clf.fit(X, y)
    tree.export_graphviz(clf, out_file="temp/"+filename+'.dot')  
    cmd = "dot -Tpng temp/"+ filename + ".dot -o pictures/" + filename + ".png"
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

    pVal = ggplot(df, aes(x='x', y='y')) + geom_point(color='lightblue') + ggtitle('Parity Chart: '+label+' | ' +"R^2: %0.4f" % r2Val ) + xlab('Experimental Value') + ylab('Predicted Value')  + stat_smooth( se=False )

    print("R^2: %0.4f" % r2Val )


    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/parity/"+file_name+".png")



class AnfisClassifier:
    'ANFIS classifier'


    def __init__(self):
        '''
            Constructor
        '''

        self.anfis = anfis.ANFIS
        self.mem = membership.membershipfunction
        self.pred = anfis.predict


    def fit(self, X, y, epochs=10):

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

    cXLog = np.log ( cX )

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

    X = np.hstack(( np.log ( np.power( X[:,[4,5]], 1.0 / X [:,[1]] ) ),  X[:,[2,3,4,5]]*X[:,[2,3,4,5]], mdea_pzratio ,  np.power( X [:,[4]], 3.0 / X [:,[5]] )   )) #log of label with R^2: 0.9108


    #Regression Based Model
    #MARS
    model = Earth()
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "MARS Model"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "Evaluate performance of 'Sweet Gas CO2 (ppm)'"
    print "X: specialformat, Y: Log"
    r2 = evaluate (model, X, y_sweetgasco2Log)
    rsquaredSweetGasCO2.append (r2)
    model.fit( X[range (ntrainingSize), :], y_sweetgasco2Log[:ntrainingSize])
    print(model.summary())

    #plot parity chart here
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgasco2Log[ntrainingSize:]
    label="Sweet Gas CO2 (MARS)"

    drawParityChart ( xdata, ydata, label )


    #ln(y) = 5698.81*ln(MDEA^(1.0/LAF)) - 15575.6*ln(PZ^(1/LAF)) - 0.000180221*LAT^2 - 0.000271076*HD^2 - 0.000810953*MDEA^2 + 0.0128298*PZ^2 - 0.378694*MDEA/PZ + 0.035448*MDEA^(3.0/PZ) + 6.85234


    print "Evaluate performance of 'Sweet Gas C1 (ppm)'"
    print "X: normal, Y: normal"
    model = Earth()
    r2 = evaluate (model, cX, y_sweetgasc1)
    rsquaredSweetGasC1.append (r2)

    model.fit( cX[range (ntrainingSize), :], y_sweetgasc1[:ntrainingSize])
    print(model.summary())

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()
    ydata = y_sweetgasc1[ntrainingSize:]
    label="Sweet Gas C1 (MARS)"

    drawParityChart ( xdata, ydata, label )


    print "Evaluate performance of 'Rich Amine Hydrocarbons (t/d)'"
    print "X: log, Y: log"
    model = Earth()
    r2 = evaluate (model, cXLog, y_richaminehydroLog)
    rsquaredRichAmineHydro.append (r2)

    model.fit( cXLog[range (ntrainingSize), :], y_richaminehydroLog[:ntrainingSize])
    print(model.summary())

    #plot parity chart here
    nxdata = cXLog[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()
    ydata = y_richaminehydroLog[ntrainingSize:]
    label="Rich Amine Hydrocarbons (MARS)"

    drawParityChart ( xdata, ydata, label )


    print "Evaluate performance of 'R Amine HCO3 (mol/L)'"
    print "X: normal, Y: normal"
    model = Earth()
    r2 = evaluate (model, cX, y_richaminehco3)
    rsquaredRichAmineHco3.append (r2)

    model.fit( cX[range (ntrainingSize), :], y_richaminehco3[:ntrainingSize] )
    print(model.summary())

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()
    ydata = y_richaminehco3[ntrainingSize:]
    label="R Amine HCO3 (MARS)"

    drawParityChart ( xdata, ydata, label )

    '''
    print "Evaluate performance of 'Sweet Gas MDEA Flow (t/d)'"
    print "X: normal, Y: log"
    model = Earth()
    r2 = evaluate (model, cX, y_sweetgasmdeaflowLog)
    rsquaredSweetGasMdeaFlow.append (r2)

    model.fit( cX[range (ntrainingSize), :], y_sweetgasmdeaflowLog[:ntrainingSize] )
    print(model.summary())

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgasmdeaflowLog[ntrainingSize:]
    label="Sweet Gas MDEA Flow (MARS)"

    drawParityChart ( xdata, ydata, label )
    '''

    print "Evaluate performance of 'Sweet Gas PZ Flow (t/d)'"
    print "X: log, Y: log"
    model = Earth()
    r2 = evaluate (model, cXLog, y_sweetgaspzflowLog)
    rsquaredSweetGaspzFlow.append (r2)

    model.fit( cXLog[range (ntrainingSize), :], y_sweetgaspzflowLog[:ntrainingSize])
    print(model.summary())

    #plot parity chart here
    nxdata = cXLog[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgaspzflowLog[ntrainingSize:]
    label="Sweet Gas PZ Flow (MARS)"

    drawParityChart ( xdata, ydata, label )


    #REC and sensitivity analysis
    model = Earth()
    #print "REC of 'Sweet Gas CO2 (ppm)'"
    cur = modelValidation (model,  X, y_sweetgasco2Log)
    print "AOC of 'Sweet Gas CO2 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasCO2.append ( cur )


    model = Earth()
    #print "REC of 'Sweet Gas C1 (ppm)'"
    cur = modelValidation (model, cX, y_sweetgasc1 )
    print "AOC of 'Sweet Gas C1 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasC1.append ( cur )


    model = Earth()
    #print "REC of 'Rich Amine Hydrocarbons (t/d)'"
    cur = modelValidation (model, cXLog, y_richaminehydroLog )
    print "AOC of 'Rich Amine Hydrocarbons (t/d)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHydro.append ( cur )

    model = Earth()
    #print "REC of 'R Amine HCO3 (mol/L)'"
    cur = modelValidation (model, cX, y_richaminehco3)
    print "AOC of 'R Amine HCO3 (mol/L)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHco3.append ( cur )

    '''
    model = Earth()
    #print "REC of Sweet Gas MDEA Flow (t/d)"
    cur = modelValidation (model, cX, y_sweetgasmdeaflowLog )
    print "AOC of 'Sweet Gas MDEA Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasMdeaFlow.append ( cur )
    '''

    model = Earth()
    #print "REC of 'Sweet Gas PZ Flow (t/d)'"
    cur = modelValidation (model, cXLog, y_sweetgaspzflowLog )
    print "AOC of 'Sweet Gas PZ Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGaspzFlow.append ( cur )



    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "Tree Model"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"


    #Tree Based Model
    clf1 = ExtraTreesRegressor(max_depth=20, n_estimators=450, random_state=10) #
    clf2 = ExtraTreesRegressor(max_depth=20, n_estimators=450, random_state=10)
    clf3 = ExtraTreesRegressor(max_depth=10, n_estimators=120, random_state=10) #
    clf4 = ExtraTreesRegressor(max_depth=15, n_estimators=150, random_state=10) #
    #clf5 = ExtraTreesRegressor(max_depth=15, n_estimators=150, random_state=10)
    clf6 = ExtraTreesRegressor(max_depth=15, n_estimators=60, random_state=10) #


    print "Evaluate performance of 'Sweet Gas CO2 (ppm)'"
    print "----------------------------------------------------------"
    currentlist = []

    r2 = evaluate (clf1, cX, y_sweetgasco2)
    rsquaredSweetGasCO2.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas CO2 (ppm)'"
    imp1 = featureImportance(clf1, df, y_sweetgasco2)
    currentlist.append (imp1)

    print imp1 


    clf1.fit( cX[range (ntrainingSize), :], y_sweetgasco2[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf1.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgasco2[ntrainingSize:]
    label="Sweet Gas CO2 (Tree)"

    drawParityChart ( xdata, ydata, label )



    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas C1 (ppm)'"
    print "----------------------------------------------------------"

    r2 = evaluate (clf2, cX, y_sweetgasc1)
    rsquaredSweetGasC1.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas C1 (ppm)'"
    imp1 = featureImportance(clf2, df, y_sweetgasc1)
    currentlist.append (imp1)

    print imp1 

    clf2.fit( cX[range (ntrainingSize), :], y_sweetgasc1[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf2.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgasc1[ntrainingSize:]
    label="Sweet Gas C1 (Tree)"

    drawParityChart ( xdata, ydata, label )

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Rich Amine Hydrocarbons (t/d)'"
    print "----------------------------------------------------------"

    r2 = evaluate (clf3, cX, y_richaminehydro)
    rsquaredRichAmineHydro.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Rich Amine Hydrocarbons (t/d)'"
    imp1 = featureImportance(clf3, df, y_richaminehydro)
    currentlist.append (imp1)

    print imp1 

    clf3.fit( cX[range (ntrainingSize), :], y_richaminehydro[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf3.predict(nxdata)
    xdata = xdata.T.tolist()
    ydata = y_richaminehydro[ntrainingSize:]
    label="Rich Amine Hydrocarbons (Tree)"

    drawParityChart ( xdata, ydata, label )

    print "----------------------------------------------------------"
    print "Evaluate performance of 'R Amine HCO3 (mol/L)'"
    print "----------------------------------------------------------"

    r2 = evaluate (clf4, cX, y_richaminehco3)
    rsquaredRichAmineHco3.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'R Amine HCO3 (mol/L)'"
    imp1 = featureImportance(clf4, df, y_richaminehco3)
    currentlist.append (imp1)

    print imp1 

    clf4.fit( cX[range (ntrainingSize), :], y_richaminehco3[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf4.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_richaminehco3[ntrainingSize:]
    label="R Amine HCO3 (Tree)"

    drawParityChart ( xdata, ydata, label )

    '''
    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas MDEA Flow (t/d)'"
    print "----------------------------------------------------------"

    r2 = evaluate (clf5, cX, y_sweetgasmdeaflow)
    rsquaredSweetGasMdeaFlow.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas MDEA Flow (t/d)'"
    imp1 = featureImportance(clf5, df, y_sweetgasmdeaflow)
    currentlist.append (imp1)

    print imp1 

    clf5.fit( cX[range (ntrainingSize), :], y_sweetgasmdeaflow[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf5.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgasmdeaflow[ntrainingSize:]
    label="Sweet Gas MDEA Flow (Tree)"

    drawParityChart ( xdata, ydata, label )
    '''

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas PZ Flow (t/d)'"
    print "----------------------------------------------------------"

    r2 = evaluate (clf6, cX, y_sweetgaspzflow)
    rsquaredSweetGaspzFlow.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas PZ Flow (t/d)'"
    imp1 = featureImportance(clf6, df, y_sweetgaspzflow)
    currentlist.append (imp1)

    print imp1 


    clf6.fit( cX[range (ntrainingSize), :], y_sweetgaspzflow[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = clf6.predict(nxdata)
    xdata = xdata.T.tolist() 
    ydata = y_sweetgaspzflow[ntrainingSize:]
    label="Sweet Gas MDEA Flow (Tree)"

    drawParityChart ( xdata, ydata, label )



    featureImpList.append ( currentlist )



    #REC and sensitivity analysis

    #print "REC of 'Sweet Gas CO2 (ppm)'"
    cur = modelValidation (clf1,  cX, y_sweetgasco2)
    print "AOC of 'Sweet Gas CO2 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasCO2.append ( cur )

    #print "REC of 'Sweet Gas C1 (ppm)'"
    cur = modelValidation (clf2, cX, y_sweetgasc1 )
    print "AOC of 'Sweet Gas C1 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasC1.append ( cur )

    #print "REC of 'Rich Amine Hydrocarbons (t/d)'"
    cur = modelValidation (clf3, cX, y_richaminehydro )
    print "AOC of 'Rich Amine Hydrocarbons (t/d)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHydro.append ( cur )

    #print "REC of 'R Amine HCO3 (mol/L)'"
    cur = modelValidation (clf4, cX, y_richaminehco3)
    print "AOC of 'R Amine HCO3 (mol/L)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHco3.append ( cur )

    '''
    #print "REC of Sweet Gas MDEA Flow (t/d)"
    cur = modelValidation (clf5, cX, y_sweetgasmdeaflow )
    print "AOC of 'Sweet Gas MDEA Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasMdeaFlow.append ( cur )
    '''

    #print "REC of Sweet Gas PZ Flow (t/d)'"
    cur = modelValidation (clf6, cX, y_sweetgaspzflow )
    print "AOC of 'Sweet Gas PZ Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGaspzFlow.append ( cur )



    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "Anfis Model"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    #Anfis Model
    model = AnfisClassifier()

    print "Evaluate performance of 'Sweet Gas CO2 (ppm)'"
    print "----------------------------------------------------------"
    currentlist = []

    r2 = crossValScore (model, cX, y_sweetgasco2)
    print "AVG R^2: " + str (r2)
    rsquaredSweetGasCO2.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas CO2 (ppm)'"
    imp1 = featureImportanceCV(model, df, y_sweetgasco2)
    currentlist.append (imp1)

    print imp1 

    model.fit( cX[range (ntrainingSize), :], y_sweetgasco2[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_sweetgasco2[ntrainingSize:]
    label="Sweet Gas CO2 (ANFIS)"

    drawParityChart ( xdata, ydata, label )



    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas C1 (ppm)'"
    print "----------------------------------------------------------"
    model = AnfisClassifier()
    r2 = crossValScore (model, cX, y_sweetgasc1)
    print "AVG R^2: " + str (r2)
    rsquaredSweetGasC1.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas C1 (ppm)'"
    imp1 = featureImportanceCV(model, df, y_sweetgasc1)
    currentlist.append (imp1)

    print imp1 


    model.fit( cX[range (ntrainingSize), :], y_sweetgasc1[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_sweetgasc1[ntrainingSize:]
    label="Sweet Gas C1 (ANFIS)"

    drawParityChart ( xdata, ydata, label )

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Rich Amine Hydrocarbons (t/d)'"
    print "----------------------------------------------------------"
    model = AnfisClassifier()
    r2 = crossValScore (model, cX, y_richaminehydro)
    print "AVG R^2: " + str (r2)
    rsquaredRichAmineHydro.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Rich Amine Hydrocarbons (t/d)'"
    imp1 = featureImportanceCV(model, df, y_richaminehydro)
    currentlist.append (imp1)

    print imp1 

    model.fit( cX[range (ntrainingSize), :], y_richaminehydro[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_richaminehydro[ntrainingSize:]
    label="Rich Amine Hydrocarbons (ANFIS)"

    drawParityChart ( xdata, ydata, label )

    print "----------------------------------------------------------"
    print "Evaluate performance of 'R Amine HCO3 (mol/L)'"
    print "----------------------------------------------------------"
    model = AnfisClassifier()
    r2 = crossValScore (model, cX, y_richaminehco3)
    print "AVG R^2: " + str (r2)
    rsquaredRichAmineHco3.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'R Amine HCO3 (mol/L)'"
    imp1 = featureImportanceCV(model, df, y_richaminehco3)
    currentlist.append (imp1)

    print imp1 

    model.fit( cX[range (ntrainingSize), :], y_richaminehco3[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_richaminehco3[ntrainingSize:]
    label="R Amine HCO3 (ANFIS)"

    drawParityChart ( xdata, ydata, label )

    '''
    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas MDEA Flow (t/d)'"
    print "----------------------------------------------------------"
    print y_sweetgasmdeaflow
    model = AnfisClassifier()
    r2 = crossValScore (model, cX, y_sweetgasmdeaflow)
    print "AVG R^2: " + str (r2)
    rsquaredSweetGasMdeaFlow.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas MDEA Flow (t/d)'"
    imp1 = featureImportanceCV(model, df, y_sweetgasmdeaflow)
    currentlist.append (imp1)

    print imp1 

    model.fit( cX[range (ntrainingSize), :], y_sweetgasmdeaflow[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_sweetgasmdeaflow[ntrainingSize:]
    label="Sweet Gas MDEA Flow (ANFIS)"

    drawParityChart ( xdata, ydata, label )
    '''


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas PZ Flow (t/d)'"
    print "----------------------------------------------------------"
    model = AnfisClassifier()
    r2 = crossValScore (model, cX, y_sweetgaspzflow)
    print "AVG R^2: " + str (r2)
    rsquaredSweetGaspzFlow.append (r2)

    #Relative importance of Variable
    print "Relative importance of Variable"
    #feature importance
    print "feature importance of 'Sweet Gas PZ Flow (t/d)'"
    imp1 = featureImportanceCV(model, df, y_sweetgaspzflow)
    currentlist.append (imp1)

    print imp1 

    model.fit( cX[range (ntrainingSize), :], y_sweetgaspzflow[:ntrainingSize])

    #plot parity chart here
    nxdata = cX[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y_sweetgaspzflow[ntrainingSize:]
    label="Sweet Gas PZ Flow (ANFIS)"

    drawParityChart ( xdata, ydata, label )

    featureImpList.append ( currentlist )


    #REC and sensitivity analysis

    #print "REC of 'Sweet Gas CO2 (ppm)'"
    model = AnfisClassifier()
    cur = modelValidationCV (model,  cX, y_sweetgasco2)
    print "AOC of 'Sweet Gas CO2 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasCO2.append ( cur )

    #print "REC of 'Sweet Gas C1 (ppm)'"
    model = AnfisClassifier()
    cur = modelValidationCV (model, cX, y_sweetgasc1 )
    print "AOC of 'Sweet Gas C1 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasC1.append ( cur )

    #print "REC of 'Rich Amine Hydrocarbons (t/d)'"
    model = AnfisClassifier()
    cur = modelValidationCV (model, cX, y_richaminehydro )
    print "AOC of 'Rich Amine Hydrocarbons (t/d)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHydro.append ( cur )

    #print "REC of 'R Amine HCO3 (mol/L)'"
    model = AnfisClassifier()
    cur = modelValidationCV (model, cX, y_richaminehco3)
    print "AOC of 'R Amine HCO3 (mol/L)': " +str (cur["avgArea"])
    recListRichAmineHco3.append ( cur )

    '''
    #print "REC of Sweet Gas MDEA Flow (t/d)"
    model = AnfisClassifier()
    cur = modelValidationCV (model, cX, y_sweetgasmdeaflow )
    print "AOC of 'Sweet Gas MDEA Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasMdeaFlow.append ( cur )
    '''
    

    #print "REC of Sweet Gas PZ Flow (t/d)'"
    model = AnfisClassifier()
    cur = modelValidationCV (model, cX, y_sweetgaspzflow )
    print "AOC of 'Sweet Gas PZ Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGaspzFlow.append ( cur )


    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "Base Model"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"


    print "Evaluate performance of 'Sweet Gas CO2 (ppm)'"
    print "----------------------------------------------------------"


    r2 = crossValNULLScore ( cX, y_sweetgasco2)
    rsquaredSweetGasCO2.append (r2)


    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas C1 (ppm)'"
    print "----------------------------------------------------------"

    r2 = crossValNULLScore ( cX, y_sweetgasc1)
    rsquaredSweetGasC1.append (r2)

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Rich Amine Hydrocarbons (t/d)'"
    print "----------------------------------------------------------"

    r2 = crossValNULLScore ( cX, y_richaminehydro)
    rsquaredRichAmineHydro.append (r2)

    print "----------------------------------------------------------"
    print "Evaluate performance of 'R Amine HCO3 (mol/L)'"
    print "----------------------------------------------------------"

    r2 = crossValNULLScore ( cX, y_richaminehco3)
    rsquaredRichAmineHco3.append (r2)

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas MDEA Flow (t/d)'"
    print "----------------------------------------------------------"
    '''
    r2 = crossValNULLScore ( cX, y_sweetgasmdeaflow)
    rsquaredSweetGasMdeaFlow.append (r2)
    '''

    print "----------------------------------------------------------"
    print "Evaluate performance of 'Sweet Gas PZ Flow (t/d)'"
    print "----------------------------------------------------------"

    r2 = crossValNULLScore ( cX, y_sweetgaspzflow)
    rsquaredSweetGaspzFlow.append (r2)

    #REC and sensitivity analysis

    #print "REC of 'Sweet Gas CO2 (ppm)'"
    cur = modelNULLValidation (  cX, y_sweetgasco2)
    print "AOC of 'Sweet Gas CO2 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasCO2.append ( cur )

    #print "REC of 'Sweet Gas C1 (ppm)'"
    cur = modelNULLValidation ( cX, y_sweetgasc1 )
    print "AOC of 'Sweet Gas C1 (ppm)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasC1.append ( cur )

    #print "REC of 'Rich Amine Hydrocarbons (t/d)'"
    cur = modelNULLValidation ( cX, y_richaminehydro )
    print "AOC of 'Rich Amine Hydrocarbons (t/d)': " +str (cur["avgArea"])
    #print cur
    recListRichAmineHydro.append ( cur )

    #print "REC of 'R Amine HCO3 (mol/L)'"
    cur = modelNULLValidation ( cX, y_richaminehco3)
    print "AOC of 'R Amine HCO3 (mol/L)': " +str (cur["avgArea"])
    recListRichAmineHco3.append ( cur )

    '''
    #print "REC of Sweet Gas MDEA Flow (t/d)"
    cur = modelNULLValidation ( cX, y_sweetgasmdeaflow )
    print "AOC of 'Sweet Gas MDEA Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGasMdeaFlow.append ( cur )
    '''

    #print "REC of Sweet Gas PZ Flow (t/d)'"
    cur = modelNULLValidation ( cX, y_sweetgaspzflow )
    print "AOC of 'Sweet Gas PZ Flow (t/d)': " +str (cur["avgArea"])
    #print cur
    recListSweetGaspzFlow.append ( cur )


    #plot the REC curve

    drawRECCURVE ( recListSweetGasCO2, rsquaredSweetGasCO2, 'Sweet Gas CO2'  )
    drawRECCURVE ( recListSweetGasC1, rsquaredSweetGasC1, 'Sweet Gas C1 ' )
    drawRECCURVE ( recListRichAmineHydro, rsquaredRichAmineHydro, 'Rich Amine Hydrocarbons' )
    drawRECCURVE ( recListRichAmineHco3, rsquaredRichAmineHco3, 'R Amine HCO3'  )
    #drawRECCURVE ( recListSweetGasMdeaFlow, rsquaredSweetGasMdeaFlow, 'Sweet Gas MDEA Flow' )
    drawRECCURVE ( recListSweetGaspzFlow, rsquaredSweetGaspzFlow, 'Sweet Gas PZ Flow' )


    #plot variable importance using histogram

    drawFeatureImportance ( featureImpList)



