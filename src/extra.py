import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error 	 


from anf import AnfisClassifier

df = pd.read_csv('data/data4.csv')


predictorLabel = df.columns[:-11].tolist()



print "List of Attributes"
print predictorLabel



print 'Evaluation metrics'
print '-------------------------------'
print 'R^2: Coefficient of Determination'
print 'MAE: Mean Absolute Error'
print 'MSE: Mean Squared Error'
print '%AARD: % Average Absolute Relative Deviation'



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
    '''
        Both input are numpy array
    '''
    absdiff = np.fabs ( y_test - y_pred).reshape ((1, len(y_pred) ))
    #diff =  np.squeeze( absdiff ) / y_test.reshape (( 1, len(y_test) ))

    diff =   absdiff / y_test.reshape (( 1, len(y_test) ))

    result = ( 100.0 / len(y_test) ) *  np.sum (np.fabs (diff) )
    return result



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





def parityChartToCSV ( xdata, ydata, label, scale = 1.0 ):
    '''
        write to CSV
    '''

    labels = ["Predicted Value", "Expected Value"]
    dfv = pd.DataFrame(columns=labels)

    xdata = [(x / scale) for x in xdata]
    ydata = [(x / scale) for x in ydata]

    for i in range(len(xdata)):
        dfv.loc[i] = [xdata[i], ydata[i]]

    #dfv[labels] = df[labels].astype(float)

    file_name = label.replace(" ", "_")
    filename = "result/csvplant/"+file_name + ".csv"
    dfv.to_csv(filename, header=True, index=False)


def unison_shuffled_copies(a, b, c, d, e, f, g, h, i, j, k):
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p], e[p], f[p], g[p], h[p], i[p], j[p], k[p]



def unison_shuffled(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]




if __name__ == "__main__":

    X = getPredictorData (df)

    y_sweetgasco2 = getLabelData (df, 'Sweet Gas CO2 (ppm)')
    y_rAmineloading = getLabelData (df, 'R Amine Loading ()')
    y_lAmineloading = getLabelData (df, 'L Amine Loading ()')
    y_sweetgasmdeaflow = getLabelData (df, 'Sweet Gas MDEA Flow (t/d)')
    y_sweetgaspzflow = getLabelData (df, 'Sweet Gas PZ Flow (t/d)')
    y_sweetgasco2flow = getLabelData (df, 'Sweet Gas CO2 Flow (t/d)')
    y_rA_c5flow = getLabelData (df, 'Rich Amine C5 Flow (t/d)')


    y_raminec5 = getLabelData (df, 'R Amine C5 (ppm)')
    y_raminec1flow = getLabelData (df, 'R Amine C1 Flow (t/d)')
    y_sweetgasc1flow = getLabelData (df, 'Sweet Gas C1 Flow (t/d)')


    y_sweetgasco2_raw = y_sweetgasco2.ravel() 
    y_rAmineloading_raw =  y_rAmineloading.ravel() 
    y_lAmineloading_raw =  y_lAmineloading.ravel() 
    y_sweetgasmdeaflow_raw =  y_sweetgasmdeaflow.ravel() 
    y_sweetgaspzflow_raw =  y_sweetgaspzflow.ravel() 
    y_sweetgasco2flow_raw =  y_sweetgasco2flow.ravel() 
    y_rA_c5flow_raw =  y_rA_c5flow.ravel() 

    y_raminec5_raw = y_raminec5.ravel()
    y_raminec1flow_raw = y_raminec1flow.ravel()
    y_sweetgasc1flow_raw = y_sweetgasc1flow.ravel()



    y_sweetgasco2 = np.log ( y_sweetgasco2_raw ) 
    y_rAmineloading =  np.log ( y_rAmineloading_raw ) 
    y_lAmineloading =  np.log ( y_lAmineloading_raw ) 
    y_sweetgasmdeaflow =  np.log ( y_sweetgasmdeaflow_raw ) 
    y_sweetgaspzflow =  np.log ( y_sweetgaspzflow_raw ) 
    y_sweetgasco2flow =  np.log ( y_sweetgasco2flow_raw ) 
    y_rA_c5flow =  np.log ( y_rA_c5flow_raw ) 

    y_raminec5 = np.log ( y_raminec5_raw )
    y_raminec1flow = np.log ( y_raminec1flow_raw )
    y_sweetgasc1flow = np.log ( y_sweetgasc1flow_raw )


    mdeaCol = X [:,[4]]
    pzCol = X [:,[5]]

    mdea_pzratio = mdeaCol /  pzCol



    cX = np.hstack((  X, mdea_pzratio ))

    #cX = np.log ( cX )

    #shuffling the data
    cX, y_sweetgasco2, y_rAmineloading, y_lAmineloading, y_sweetgasmdeaflow, y_sweetgaspzflow, y_sweetgasco2flow, y_rA_c5flow, y_raminec5, y_raminec1flow, y_sweetgasc1flow = unison_shuffled_copies(cX, y_sweetgasco2, y_rAmineloading, y_lAmineloading, y_sweetgasmdeaflow, y_sweetgaspzflow, y_sweetgasco2flow, y_rA_c5flow, y_raminec5, y_raminec1flow, y_sweetgasc1flow  )

    

    ntrainingSize = int (0.7 * len ( cX )) # 70 - 30 split

    #zero mean for ANFIS model

    '''
    meanList = cX.mean(axis=0)
    meanMatrixlist = []
    for i in range(len(cX)):
        mean_line = meanList.reshape(1,len(meanList))
        meanMatrixlist.append(mean_line)

    meanMatrix = np.vstack(meanMatrixlist)

    zero_meanX = cX - meanMatrix
    '''
    minArr = np.amin( cX, axis=0 )
    maxArr  = np.amax( cX, axis=0 )


    minMatrixlist = []
    maxMatrixlist = []
    for i in range(len(cX)):
        min_line = minArr.reshape(1,len(minArr))
        minMatrixlist.append(min_line)

        max_line = maxArr.reshape(1,len(maxArr))
        maxMatrixlist.append(max_line)

    minMatrix = np.vstack(minMatrixlist)
    maxMatrix = np.vstack(maxMatrixlist)



    zero_meanX = 0.1 + ( 0.8 * ( cX - minMatrix ) / (maxMatrix - minMatrix) )



    print "Evaluation of SG CO2 (ppm)\n"


    model = AnfisClassifier()
    model.setType ('gaussmf')

    r2 = crossValScore (model, zero_meanX, y_sweetgasco2)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_sweetgasco2 )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="SG CO2 (ppm) (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="SG CO2 (ppm) (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )



    print "Evaluation of RAL ()\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_rAmineloading)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_rAmineloading )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="RAL (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="RAL (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    print "Evaluation of LAL ()\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_lAmineloading)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_lAmineloading )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="LAL (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )

    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="LAL (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )



    print "Evaluation of SG MDEA Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')

    scaleVal = 1000000000000.0
    r2 = crossValScore (model, zero_meanX, scaleVal * y_sweetgasmdeaflow, scale = scaleVal)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, scaleVal * y_sweetgasmdeaflow )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="SG MDEA Flow (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label, scale = scaleVal )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="SG MDEA Flow (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label, scale = scaleVal )


    print "Evaluation of SG PZ Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')

    r2 = crossValScore (model, zero_meanX, y_sweetgaspzflow)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_sweetgaspzflow )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="SG PZ Flow (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="SG PZ Flow (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    print "Evaluation of SG CO2 Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_sweetgasco2flow)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_sweetgasco2flow )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )

    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]


    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="SG CO2 Flow (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="SG CO2 Flow (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    print "Evaluation of RA C5 Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_rA_c5flow)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_rA_c5flow )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="RA C5 Flow (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="RA C5 Flow (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )



    print "Evaluation of R Amine C5 (ppm)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_raminec5)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_raminec5 )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="R Amine C5 (ppm) (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="R Amine C5 (ppm) (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    print "Evaluation of R Amine C1 Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')


    r2 = crossValScore (model, zero_meanX, y_raminec1flow )
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, y_raminec1flow  )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="R Amine C1 Flow (td) (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="R Amine C1 Flow (td) (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label )
 

    print "Evaluation of Sweet Gas C1 Flow (t/d)\n"

    model = AnfisClassifier()
    model.setType ('gaussmf')
    scaleVal = 1000000000000.0

    r2 = crossValScore (model, zero_meanX, scaleVal * y_sweetgasc1flow_raw, scale = scaleVal)
    print r2

    model = AnfisClassifier()
    model.setType ('gaussmf')
    X,y = unison_shuffled( zero_meanX, scaleVal * y_sweetgasc1flow_raw  )
    model.fit( X[range (ntrainingSize), :], y[:ntrainingSize] )


    #plot parity chart here (training)
    nxdata = X[range (ntrainingSize), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[:ntrainingSize]
    label="Sweet Gas C1 Flow (td) (training)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label, scale = scaleVal )


    #plot parity chart here (testing)
    nxdata = X[range (ntrainingSize, len(X)), :]
    xdata = model.predict(nxdata)
    xdata = xdata.T.tolist()[0]
    ydata = y[ntrainingSize:]
    label="Sweet Gas C1 Flow (td) (testing)(ANFIS)"

    parityChartToCSV ( xdata, ydata, label, scale = scaleVal )



