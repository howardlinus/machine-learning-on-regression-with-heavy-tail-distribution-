from scipy.stats import cauchy
import numpy as np
class Cauchy_regression1(object):
    # Define classs parameters
    # model noise with c
    c = np.random.random()
    d = np.random.random()
    
    def __init__(self, lambd=0):                
        #Weights (parameters)
        self.a = np.random.random()
        self.b = np.random.random()
            
        #Regularization Parameter:
        self.lambd = lambd
                    
            
    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        c=Cauchy_regression1.c
        d=Cauchy_regression1.d
        gamma = abs(c*x + d)
        yHat = self.a*x+self.b
        J = -   sum(np.log(gamma))/x.shape[0] - sum(np.log((y-yHat)**2 + gamma**2))/x.shape[0] + (self.lambd/2)*(sum(self.getParams()**2))
        
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d for a given x and y:
        c=Cauchy_regression1.c
        d=Cauchy_regression1.d
        gamma = (c*x + d)
        yHat = self.a*x+self.b
        dJda = sum(x*2*(y-yHat)/(gamma**2+(y-yHat)**2))/x.shape[0] + self.lambd*self.a
        dJdb = sum(2*(y-yHat)/(gamma**2+(y-yHat)**2))/x.shape[0] + self.lambd*self.b
        dJdc = -   sum(x/gamma)/x.shape[0] - sum(x*2*gamma/(gamma**2+(y-yHat)**2))/x.shape[0] + self.lambd*c
        dJdd = -   sum(1/gamma)/x.shape[0] - sum(2*gamma/(gamma**2+(y-yHat)**2))/x.shape[0] + self.lambd*d       
        
        return dJda, dJdb, dJdc, dJdd

    def computeGradients(self, x, y, L=0):       
        dJda, dJdb, dJdc, dJdd = self.costFunctionPrime(x, y)
        if L==1:
            return np.asarray((dJdc, dJdd))
        elif L==2:
            return np.asarray((dJda, dJdb))
        else:
            return np.asarray((dJda, dJdb, dJdc, dJdd))
    
    def getParams(self, L=0):
        #Get a, b, c, d :
        if L==1:
            params = np.asarray((Cauchy_regression1.c, Cauchy_regression1.d))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Cauchy_regression1.c, Cauchy_regression1.d))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d :
        if L==1:
            Cauchy_regression1.c = params[0]
            Cauchy_regression1.d = params[1]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Cauchy_regression1.c = params[2]
            Cauchy_regression1.d = params[3]
        
    def computeNumericalGradients(self, x, y):
        paramsInitial = self.getParams()
        numgrad = np.zeros(len(paramsInitial))
        perturb = np.zeros(len(paramsInitial))        
        e = 1e-5
        
        for n in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[n] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costFunction(x, y)
            
            self.setParams(paramsInitial - perturb)
            loss1 = self.costFunction(x, y)
            
            #Compute Numerical Gradient
            numgrad[n] = (loss2 - loss1) / (2*e)
            
            #Return the value we changed to zero:
            perturb[n] = 0

        #Return Params to original value:
        self.setParams(paramsInitial)

        return numgrad 
    
    def dataGeneration(self, x1=-5, x2=5, N=100):
        c=Cauchy_regression1.c
        d=Cauchy_regression1.d

        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        s_rand=np.zeros(N)
        for i in range(len(x)):
            gamma =abs(c*x[i] + d)
            s_rand[i]=cauchy.rvs(loc=0, scale=gamma, size=1)
        y = self.a*x + self.b + s_rand        
        
        return np.append(x, y).reshape(2,N).transpose()
    