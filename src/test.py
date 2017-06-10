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
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error 	 

from sklearn.model_selection import train_test_split

import ggplot 
from ggplot import *
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

from anf import AnfisClassifier

'''
See paper below

Model comparison is done using Regression Error Characteristics Curves as described in Jinbo Bi, Kristin Bennett. Regression error characteristic curves. In Proceedings of the 20th international conference on machine learning, pages 43--50, 2003.
'''

df = pd.read_csv('data/data1.csv')

df2 = pd.read_csv('data/data2.csv')


predictorLabel = df.columns[:-8].tolist()

#['Lean Amine Pressure (bar)', 'Lean Amine Circulation Rate (t/d)', 'Lean Amine Temperature (C)', 'Heat Duty (GJ/h)', 'Reboiler Pressure (bar (g))', 'MDEA Concentration (%)', 'PZ Concentration (%)']



#['LP', 'LC', 'LT', 'HD', 'RP', 'MC', 'PZ']



print "List of Attributes"
print predictorLabel

print 'Evaluation metrics'
print '-------------------------------'
print 'R^2: Coefficient of Determination'
print 'MAE: Mean Absolute Error'
print 'MSE: Mean Squared Error'
print '%AARD: % Average Absolute Relative Deviation'



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


def getAARD ( y_test, y_pred ):
    """
        Both input are numpy array
    """
    absdiff = np.fabs ( y_test - y_pred).reshape ((1, len(y_pred) ))
    diff =   absdiff  / y_test.reshape (( 1, len(y_test) ))

    result = ( 100.0 / len(y_test) ) *  np.sum (diff)
    return result



def evaluate (model, X, y, n_folds=5, scale = 1.0):
    '''
        get the R2 of an ensemble on cross validation
    '''
    R2Output = []
    MAEOutput = []
    MSEOutput = []
    AARDOutput = []

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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        y_pred = y_pred.reshape ((len(X_test),1))
        y_test = y_test.reshape ((len(X_test),1))


        r2Val = r2_score( y_test, y_pred )
        R2Output.append (r2Val)

        maeVal = mean_absolute_error( y_test, y_pred )
        MAEOutput.append (maeVal)

        mseVal = mean_squared_error( y_test, y_pred )
        MSEOutput.append (mseVal)

        aardVal = getAARD ( y_test, y_pred )
        AARDOutput.append (aardVal)

    resultR2 = sum (R2Output) / len (R2Output) 
    resultMAE = sum (MAEOutput) / len (MAEOutput) 
    resultMSE = sum (MSEOutput) / len (MSEOutput) 
    resultAARD = sum (AARDOutput) / len (AARDOutput) 

    dictVal = { "R^2": round (resultR2, 4), "MAE": round ( (resultMAE / scale), 4 ), "MSE": round ( (resultMSE / scale**2), 4), "AARD": round (resultAARD, 4) }

    return dictVal


def evalEnsembleModel (classifierlist, X, y, n_folds=5):
    '''
        get the R2 of an ensemble on cross validation
    '''
    R2Output = []
    MAEOutput = []
    MSEOutput = []
    AARDOutput = []

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
        R2Output.append (r2Val)

        maeVal = mean_absolute_error(y_test, avgOutput)
        MAEOutput.append (maeVal)

        mseVal = mean_squared_error(y_test, avgOutput)
        MSEOutput.append (mseVal)

        aardVal = getAARD ( y_test, y_pred )
        AARDOutput.append (aardVal)

    resultR2 = sum (R2Output) / len (R2Output) 
    resultMAE = sum (MAEOutput) / len (MAEOutput) 
    resultMSE = sum (MSEOutput) / len (MSEOutput) 
    resultAARD = sum (AARDOutput) / len (AARDOutput) 

    dictVal = { "R^2": round (resultR2, 4), "MAE": round ( (resultMAE / scale), 4 ), "MSE": round ( (resultMSE / scale**2), 4), "AARD": round (resultAARD, 4) }
    return dictVal



def crossValScore (cls, X, y, n_folds=5, scale = 1.0):
    '''
        get the R2 of an ensemble on cross validation
    '''
    R2Output = []
    MAEOutput = []
    MSEOutput = []
    AARDOutput = []

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
        R2Output.append (r2Val)

        maeVal = mean_absolute_error( y_test, y_pred )
        MAEOutput.append (maeVal)

        mseVal = mean_squared_error( y_test, y_pred )
        MSEOutput.append (mseVal)

        aardVal = getAARD ( y_test, y_pred )
        AARDOutput.append (aardVal)


    resultR2 = sum (R2Output) / len (R2Output) 
    resultMAE = sum (MAEOutput) / len (MAEOutput) 
    resultMSE = sum (MSEOutput) / len (MSEOutput) 
    resultAARD = sum (AARDOutput) / len (AARDOutput) 

    dictVal = { "R^2": round (resultR2, 4), "MAE": round ( (resultMAE / scale), 4 ), "MSE": round ( (resultMSE / scale**2), 4), "AARD": round (resultAARD, 4) }
    return dictVal



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




def modelValidationCV (clf, X, y, scale = 1.0  ):
    '''
        
        return list of area of the Cv, average x, y of REC curve
    '''
    dictVal = crossValRECCV(clf, X, y )
    xlistOfLists =  dictVal["x"] # absolute error


    ylistOfLists =  dictVal["y"]
    areaList = []
    for xlist, ylist in zip (xlistOfLists, ylistOfLists):
        xlist = [(x / scale) for x in xlist] #scale the error
        aucVal = AUC( xlist, ylist )
        #Calculate the area over the curve
        xlistSort = sorted (xlist)

        totalAreaOfBox = xlistSort[-1]
        aocVal = totalAreaOfBox - aucVal
        areaList.append ( aocVal )

    xAvgList =  map(mean, zip(*xlistOfLists))
    yAvgList =  map(mean, zip(*ylistOfLists))

    #apply the scaling on the y axis

    #yAvgList = [(x / scale) for x in yAvgList]


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



def crossValNULLScore ( X, y, n_folds=5, scale=1.0 ):

    R2Output = []
    MAEOutput = []
    MSEOutput = []
    AARDOutput = []

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
        R2Output.append (r2Val)

        maeVal = mean_absolute_error(y_test, y_pred)
        MAEOutput.append (maeVal)

        mseVal = mean_squared_error(y_test, y_pred)
        MSEOutput.append (mseVal)

        aardVal = getAARD ( y_test, y_pred )
        AARDOutput.append (aardVal)

    resultR2 = sum (R2Output) / len (R2Output) 
    resultMAE = sum (MAEOutput) / len (MAEOutput) 
    resultMSE = sum (MSEOutput) / len (MSEOutput) 
    resultAARD = sum (AARDOutput) / len (AARDOutput) 

    dictVal = { "R^2": round (resultR2, 4), "MAE": round ( (resultMAE / scale), 4 ), "MSE": round ( (resultMSE / scale**2), 4), "AARD": round (resultAARD, 4) }
    return dictVal


def drawRECCURVE ( recObjectList,  label  ):
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

        legend = clsLabel[ind]
        dfA['Classifier'] = [legend]*len( rec['yAvgList'] )
        df = df.append(dfA, ignore_index=True)


    p = ggplot(df, aes(x='x', y='y', color='Classifier', group='group'))  + theme_bw()



    nullModel = recObjectList[-1]
    xMax = nullModel['xAvgList'][-1]


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


def parityChartToCSV ( xdata, ydata, label, scale = 1.0 ):
    """
        write to CSV
    """

    labels = ["Predicted Value", "Expected Value"]
    dfv = pd.DataFrame(columns=labels)

    xdata = [(x / scale) for x in xdata]
    ydata = [(x / scale) for x in ydata]

    for i in range(len(xdata)):
        dfv.loc[i] = [xdata[i], ydata[i]]

    #dfv[labels] = df[labels].astype(float)

    file_name = label.replace(" ", "_")
    filename = "result/csv/"+file_name + ".csv"
    dfv.to_csv(filename, header=True, index=False)



def drawParityChart ( xdata, ydata, label, scale = 1.0 ):
    '''
        plot parity chart
    '''

    r2Val = r2_score(  xdata, ydata )
    maeVal = mean_absolute_error(  xdata, ydata )
    maeVal = maeVal / scale

    mseVal = mean_squared_error(  xdata, ydata )
    mseVal = mseVal / scale**2


    xdatanpArr = np.asarray(xdata)
    ydatanpArr = np.asarray(ydata)

    aardVal = getAARD ( xdatanpArr, ydatanpArr)

    df = pd.DataFrame()
    xdata = [(x / scale) for x in xdata]
    ydata = [(x / scale) for x in ydata]

    df['x'] = xdata
    df['y'] = ydata





    print 'Parity Chart: {0} | R^2: {1} | MAE: {2} | MSE: {3} | AARD: {4}'.format( label, round(r2Val, 3), round(maeVal, 3), round(mseVal, 3), round(aardVal, 3))


    pVal = ggplot(df, aes(x='x', y='y')) + geom_point(color='blue') + ggtitle( 'Parity Chart: {0} | R^2: {1} | MAE: {2} | MSE: {3} | AARD: {4}'.format( label, round(r2Val, 3), round(maeVal, 3), round(mseVal, 3), round(aardVal, 3))  ) + xlab('Experimental Value') + ylab('Predicted Value')  + stat_smooth( se=False )  + theme_bw() + scale_x_continuous( limits=( min(xdata)- 0.000000005,max(xdata)+0.000000005 ) ) + scale_y_continuous( limits=(min(xdata)- 0.000000005, max(ydata)+0.000000005 ) )



    file_name = label.replace(" ", "_")
    ggplot.save(pVal, "pictures/parity/"+file_name+".png")



def unison_shuffled_copies(a, b, c, d, e, f, g, h, i):
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p], e[p], f[p], g[p], h[p], i[p]



def unison_shuffled(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]




if __name__ == "__main__":

    X = getPredictorData (df)

    y_sweetgasco2 = getLabelData (df, 'Sweet Gas CO2 (ppm)')
    y_sweetgasc1 = getLabelData (df, 'Sweet Gas C1 (ppm)')
    y_richaminehydro = getLabelData (df, 'Rich Amine Hydrocarbons (t/d)')

    y_richaminehco3 = getLabelData (df, 'R Amine HCO3 (mol/L)')
    y_sweetgasmdeaflow = getLabelData (df, 'Sweet Gas MDEA Flow (t/d)')
    y_sweetgaspzflow = getLabelData (df, 'Sweet Gas PZ Flow (t/d)')

    y_rAmineloading = getLabelData (df2, 'R Amine Loading')
    y_lAmineloading = getLabelData (df2, 'L Amine Loading')


    y_sweetgasco2 = y_sweetgasco2.ravel() 
    y_sweetgasc1 = y_sweetgasc1.ravel() 
    y_richaminehydro = y_richaminehydro.ravel() 

    y_richaminehco3 = y_richaminehco3.ravel() 
    y_sweetgasmdeaflow = y_sweetgasmdeaflow.ravel() 
    y_sweetgaspzflow = y_sweetgaspzflow.ravel() 

    y_rAmineloading = y_rAmineloading.ravel()
    y_lAmineloading = y_rAmineloading.ravel()


    mdeaCol = X [:,[4]]
    pzCol = X [:,[5]]

    mdea_pzratio = mdeaCol /  pzCol


    cX = np.hstack((  X, mdea_pzratio ))


    ntrainingSize = int (0.7 * len ( cX )) # 70 - 30 split

    #shuffling the data
    cX, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow, y_rAmineloading , y_lAmineloading  = unison_shuffled_copies( cX, y_sweetgasco2, y_sweetgasc1, y_richaminehydro, y_richaminehco3, y_sweetgasmdeaflow, y_sweetgaspzflow, y_rAmineloading , y_lAmineloading  )



    #zero mean for ANFIS model


    meanList = cX.mean(axis=0)
    meanMatrixlist = []
    for i in range(len(cX)):
        mean_line = meanList.reshape(1,len(meanList))
        meanMatrixlist.append(mean_line)

    meanMatrix = np.vstack(meanMatrixlist)

    zero_meanX = cX - meanMatrix

    stdList = cX.std(axis=0) 

    zero_meanX = zero_meanX / stdList #standardization






    #Regression Based Model
    #MARS
    model = Earth()
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "MARS Model"
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"

    print "Evaluate performance of 'Sweet Gas CO2 (ppm)'"
    r2 = evaluate (model, zero_meanX, y_sweetgasco2)
    print r2

    model = Earth()

    X,y = unison_shuffled(zero_meanX, y_sweetgasco2)
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize])
    print(model.summary())

  
