import numpy as np
from scipy import optimize

class trainer(object):
    def __init__(self, Reg):
        #Make Local reference to regression class:
        self.Reg = Reg
        
    def callbackF(self, params):
        self.Reg.setParams(params)
        self.J.append(self.Reg.costFunction(self.x, self.y))
        self.testJ.append(self.Reg.costFunction(self.testX, self.testY))

    def callbackFL1(self, params):
        self.Reg.setParams(params, L=1)
        self.J.append(self.Reg.costFunction(self.x, self.y))
        self.testJ.append(self.Reg.costFunction(self.testX, self.testY))
        
    def callbackFL2(self, params):
        self.Reg.setParams(params, L=2)
        self.J.append(self.Reg.costFunction(self.x, self.y))
        self.testJ.append(self.Reg.costFunction(self.testX, self.testY))
        
    def costFunctionWrapper(self, params, x, y):
        self.Reg.setParams(params)
        cost = self.Reg.costFunction(x, y)
        grad = self.Reg.computeGradients(x,y)
        
        return cost, grad
        
    def costFunctionWrapperL1(self, params, x, y):
        self.Reg.setParams(params, L=1)
        cost = self.Reg.costFunction(x, y)
        grad = self.Reg.computeGradients(x,y,L=1)
        
        return cost, grad

    def costFunctionWrapperL2(self, params, x, y):
        self.Reg.setParams(params, L=2)
        cost = self.Reg.costFunction(x, y)
        grad = self.Reg.computeGradients(x,y,L=2)
        
        return cost, grad

    def train(self, trainX, trainY, testX, testY, L=0):
        #Make an internal variable for the callback function:
        self.x = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.J = []
        self.testJ = []
        
        if L==1:
            params0 = self.Reg.getParams(L=1)
            options = {'maxiter': 200, 'disp' : True}
            _res = optimize.minimize(self.costFunctionWrapperL1, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackFL1)
            self.Reg.setParams(_res.x, L=1)
        elif L==2:
            params0 = self.Reg.getParams(L=2)
            options = {'maxiter': 200, 'disp' : True}
            _res = optimize.minimize(self.costFunctionWrapperL2, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackFL2)
            self.Reg.setParams(_res.x, L=2)
        else:    
            params0 = self.Reg.getParams()
            options = {'maxiter': 200, 'disp' : True}
            _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)
            self.Reg.setParams(_res.x)
            
        self.optimizationResults = _res


