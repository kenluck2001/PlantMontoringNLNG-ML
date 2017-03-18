from __future__ import division
from anfis import anfis
from anfis import membership #import membershipfunction, mfDerivs





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



