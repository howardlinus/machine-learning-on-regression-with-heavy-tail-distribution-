import numpy as np

# Regression Class 1:
class Heavy_tail_regression1(object):
    # Define classs parameters
    # model noise with c*x + d
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
        #sigma_2 = (self.c*x)**2 + self.d**2
        c=Heavy_tail_regression1.c
        d=Heavy_tail_regression1.d
        sigma_2 = (c*x)**2 + d**2
        yHat = self.a*x+self.b
        J = 0.5*sum((y-yHat)**2/sigma_2)/x.shape[0] + 0.5*sum(np.log(sigma_2))/x.shape[0]+(self.lambd/2)*(sum(self.getParams()**2))
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d for a given x and y:
        #sigma_2 = (self.c*x)**2 + self.d**2
        c=Heavy_tail_regression1.c
        d=Heavy_tail_regression1.d
        sigma_2 = (c*x)**2 + d**2
        yHat = self.a*x+self.b
        dJda = sum(-x*(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.a
        dJdb = sum(-(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.b
        dJdc = sum(c*x**2/sigma_2)/x.shape[0]-sum(c*x**2*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*c
        dJdd = sum(d/sigma_2)/x.shape[0]-sum(d*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*d       
        
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
            params = np.asarray((Heavy_tail_regression1.c, Heavy_tail_regression1.d))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Heavy_tail_regression1.c, Heavy_tail_regression1.d))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d :
        if L==1:
            Heavy_tail_regression1.c = params[0]
            Heavy_tail_regression1.d = params[1]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Heavy_tail_regression1.c = params[2]
            Heavy_tail_regression1.d = params[3]
        
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
        c=abs(Heavy_tail_regression1.c)
        d=abs(Heavy_tail_regression1.d)
                
        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        c_randn=np.random.normal(0, c, N)
        d_randn=np.random.normal(0, d, N)
        y = self.a*x + self.b + c_randn*x + d_randn        
        
        return np.append(x, y).reshape(2,N).transpose()
      
    
# Regression Class 2:
class Heavy_tail_regression2(object):
    # model noise with c*x**2 + d*x + e
    c = np.random.random()
    d = np.random.random()
    e = np.random.random()
    
    def __init__(self, lambd=0):        
        #Weights (parameters)
        self.a = np.random.random()
        self.b = np.random.random()
            
        #Regularization Parameter:
        self.lambd = lambd                    
            
    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        c=Heavy_tail_regression2.c
        d=Heavy_tail_regression2.d
        e=Heavy_tail_regression2.e
        sigma_2 = (c*x**2)**2 + (d*x)**2 + e**2
        yHat = self.a*x+self.b
        J = 0.5*sum((y-yHat)**2/sigma_2)/x.shape[0] + 0.5*sum(np.log(sigma_2))/x.shape[0]+(self.lambd/2)*(sum(self.getParams()**2))
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d for a given x and y:
        c=Heavy_tail_regression2.c
        d=Heavy_tail_regression2.d
        e=Heavy_tail_regression2.e
        sigma_2 = (c*x**2)**2 + (d*x)**2 + e**2
        yHat = self.a*x+self.b
        dJda = sum(-x*(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.a
        dJdb = sum(-(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.b
        dJdc = sum(c*x**4/sigma_2)/x.shape[0]-sum(c*x**4*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*c
        dJdd = sum(d*x**2/sigma_2)/x.shape[0]-sum(d*x**2*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*d
        dJde = sum(e/sigma_2)/x.shape[0]-sum(e*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*e       
        
        return dJda, dJdb, dJdc, dJdd, dJde

    def computeGradients(self, x, y, L=0):       
        dJda, dJdb, dJdc, dJdd, dJde = self.costFunctionPrime(x, y)
        if L==1:
            return np.asarray((dJdc, dJdd, dJde))
        elif L==2:
            return np.asarray((dJda, dJdb))
        else:
            return np.asarray((dJda, dJdb, dJdc, dJdd, dJde))
    
    def getParams(self, L=0):
        #Get a, b, c, d :
        if L==1:
            params = np.asarray((Heavy_tail_regression2.c, Heavy_tail_regression2.d, Heavy_tail_regression2.e))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Heavy_tail_regression2.c, Heavy_tail_regression2.d, Heavy_tail_regression2.e))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d :
        if L==1:
            Heavy_tail_regression2.c = params[0]
            Heavy_tail_regression2.d = params[1]
            Heavy_tail_regression2.e = params[2]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Heavy_tail_regression2.c = params[2]
            Heavy_tail_regression2.d = params[3]
            Heavy_tail_regression2.e = params[4]
            
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
        c=abs(Heavy_tail_regression2.c)
        d=abs(Heavy_tail_regression2.d)
        e=abs(Heavy_tail_regression2.e)

        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        c_randn=np.random.normal(0,c,N)
        d_randn=np.random.normal(0,d,N)
        e_randn=np.random.normal(0,e,N)
        y = self.a*x + self.b + c_randn*x**2 + d_randn*x + e_randn        
        
        return np.append(x, y).reshape(2,N).transpose()

# Regression Class 3:
class Heavy_tail_regression3(object):
    # model noise with c*x**3 + d*x**2 + e*x + f
    c = np.random.random()
    d = np.random.random()
    e = np.random.random()
    f = np.random.random()
    
    def __init__(self, lambd=0):        
        #Weights (parameters)
        self.a = np.random.random()
        self.b = np.random.random()
            
        #Regularization Parameter:
        self.lambd = lambd
                    
            
    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        c=Heavy_tail_regression3.c
        d=Heavy_tail_regression3.d
        e=Heavy_tail_regression3.e
        f=Heavy_tail_regression3.f
        sigma_2 = (c*x**3)**2 + (d*x**2)**2 + (e*x)**2 + f**2
        yHat = self.a*x+self.b
        J = 0.5*sum((y-yHat)**2/sigma_2)/x.shape[0] + 0.5*sum(np.log(sigma_2))/x.shape[0]+(self.lambd/2)*(sum(self.getParams()**2))
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d, e, f for a given x and y:
        c=Heavy_tail_regression3.c
        d=Heavy_tail_regression3.d
        e=Heavy_tail_regression3.e
        f=Heavy_tail_regression3.f
        sigma_2 = (c*x**3)**2 + (d*x**2)**2 + (e*x)**2 + f**2
        yHat = self.a*x+self.b
        dJda = sum(-x*(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.a
        dJdb = sum(-(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.b
        dJdc = sum(c*x**6/sigma_2)/x.shape[0]-sum(c*x**6*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*c        
        dJdd = sum(d*x**4/sigma_2)/x.shape[0]-sum(d*x**4*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*d
        dJde = sum(e*x**2/sigma_2)/x.shape[0]-sum(e*x**2*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*e
        dJdf = sum(f/sigma_2)/x.shape[0]-sum(f*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*f       
        
        return dJda, dJdb, dJdc, dJdd, dJde, dJdf

    def computeGradients(self, x, y, L=0):       
        dJda, dJdb, dJdc, dJdd, dJde, dJdf = self.costFunctionPrime(x, y)
        if L==1:
            return np.asarray((dJdc, dJdd, dJde, dJdf))
        elif L==2:
            return np.asarray((dJda, dJdb))
        else:
            return np.asarray((dJda, dJdb, dJdc, dJdd, dJde, dJdf))
    
    def getParams(self, L=0):
        #Get a, b, c, d :
        if L==1:
            params = np.asarray((Heavy_tail_regression3.c, Heavy_tail_regression3.d, Heavy_tail_regression3.e, Heavy_tail_regression3.f))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Heavy_tail_regression3.c, Heavy_tail_regression3.d, Heavy_tail_regression3.e, Heavy_tail_regression3.f))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d :
        if L==1:
            Heavy_tail_regression3.c = params[0]
            Heavy_tail_regression3.d = params[1]
            Heavy_tail_regression3.e = params[2]
            Heavy_tail_regression3.f = params[3]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Heavy_tail_regression3.c = params[2]
            Heavy_tail_regression3.d = params[3]
            Heavy_tail_regression3.e = params[4]
            Heavy_tail_regression3.f = params[5]
            
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
        c=abs(Heavy_tail_regression3.c)
        d=abs(Heavy_tail_regression3.d)
        e=abs(Heavy_tail_regression3.e)
        f=abs(Heavy_tail_regression3.f)

        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        c_randn=np.random.normal(0,c,N)
        d_randn=np.random.normal(0,d,N)
        e_randn=np.random.normal(0,e,N)
        f_randn=np.random.normal(0,f,N)       
        y = self.a*x + self.b + c_randn*x**3 + d_randn*x**2 + e_randn*x + f_randn        
        
        return np.append(x, y).reshape(2,N).transpose()
    
# Regression Class 4:
class Heavy_tail_regression4(object):
    # model noise with c*x**4 + d*x**3 + e*x**2 + f*x + g
    c = np.random.random()
    d = np.random.random()
    e = np.random.random()
    f = np.random.random()
    g = np.random.random()
    
    def __init__(self, lambd=0):        
        
        #Weights (parameters)
        self.a = np.random.random()
        self.b = np.random.random()
            
        #Regularization Parameter:
        self.lambd = lambd
                    
            
    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        c=Heavy_tail_regression4.c
        d=Heavy_tail_regression4.d
        e=Heavy_tail_regression4.e
        f=Heavy_tail_regression4.f
        g=Heavy_tail_regression4.g
        sigma_2 = (c*x**4)**2 +(d*x**3)**2 + (e*x**2)**2 + (f*x)**2 + g**2
        yHat = self.a*x+self.b
        J = 0.5*sum((y-yHat)**2/sigma_2)/x.shape[0] + 0.5*sum(np.log(sigma_2))/x.shape[0]+(self.lambd/2)*(sum(self.getParams()**2))
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d, e, f, g for a given x and y:
        c=Heavy_tail_regression4.c
        d=Heavy_tail_regression4.d
        e=Heavy_tail_regression4.e
        f=Heavy_tail_regression4.f
        g=Heavy_tail_regression4.g
        sigma_2 = (c*x**4)**2 +(d*x**3)**2 + (e*x**2)**2 + (f*x)**2 + g**2
        yHat = self.a*x+self.b
        dJda = sum(-x*(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.a
        dJdb = sum(-(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.b
        dJdc = sum(c*x**8/sigma_2)/x.shape[0]-sum(c*x**8*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*c        
        dJdd = sum(d*x**6/sigma_2)/x.shape[0]-sum(d*x**6*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*d        
        dJde = sum(e*x**4/sigma_2)/x.shape[0]-sum(e*x**4*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*e
        dJdf = sum(f*x**2/sigma_2)/x.shape[0]-sum(f*x**2*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*f
        dJdg = sum(g/sigma_2)/x.shape[0]-sum(g*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*g       
        
        return dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg

    def computeGradients(self, x, y, L=0):       
        dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg = self.costFunctionPrime(x, y)
        if L==1:
            return np.asarray((dJdc, dJdd, dJde, dJdf, dJdg))
        elif L==2:
            return np.asarray((dJda, dJdb))
        else:
            return np.asarray((dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg))
    
    def getParams(self, L=0):
        #Get a, b, c, d :
        if L==1:
            params = np.asarray((Heavy_tail_regression4.c, Heavy_tail_regression4.d, Heavy_tail_regression4.e, Heavy_tail_regression4.f, Heavy_tail_regression4.g))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Heavy_tail_regression4.c, Heavy_tail_regression4.d, Heavy_tail_regression4.e, Heavy_tail_regression4.f, Heavy_tail_regression4.g))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d :
        if L==1:
            Heavy_tail_regression4.c = params[0]
            Heavy_tail_regression4.d = params[1]
            Heavy_tail_regression4.e = params[2]
            Heavy_tail_regression4.f = params[3]
            Heavy_tail_regression4.g = params[4]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Heavy_tail_regression4.c = params[2]
            Heavy_tail_regression4.d = params[3]
            Heavy_tail_regression4.e = params[4]
            Heavy_tail_regression4.f = params[5]
            Heavy_tail_regression4.g = params[6]
            
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
        c=abs(Heavy_tail_regression4.c)
        d=abs(Heavy_tail_regression4.d)
        e=abs(Heavy_tail_regression4.e)
        f=abs(Heavy_tail_regression4.f)
        g=abs(Heavy_tail_regression4.g)

        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        c_randn=np.random.normal(0,c,N)
        d_randn=np.random.normal(0,d,N)
        e_randn=np.random.normal(0,e,N)
        f_randn=np.random.normal(0,f,N)
        g_randn=np.random.normal(0,g,N)
        y = self.a*x + self.b + c_randn*x**4 + d_randn*x**3 + e_randn*x**2 + f_randn*x + g_randn        
        
        return np.append(x, y).reshape(2,N).transpose()
    
# Regression Class 5:
class Heavy_tail_regression5(object):
    # model noise with c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    c = np.random.random()
    d = np.random.random()
    e = np.random.random()
    f = np.random.random()
    g = np.random.random()
    h = np.random.random()
    
    def __init__(self, lambd=0):        
        
        #Weights (parameters)
        self.a = np.random.random()
        self.b = np.random.random()
            
        #Regularization Parameter:
        self.lambd = lambd
                    
            
    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        #sigma_2 = (self.c*x)**2 + self.d**2
        c=Heavy_tail_regression5.c
        d=Heavy_tail_regression5.d
        e=Heavy_tail_regression5.e
        f=Heavy_tail_regression5.f
        g=Heavy_tail_regression5.g
        h=Heavy_tail_regression5.h
        sigma_2 = (c*x**5)**2 + (d*x**4)**2 +(e*x**3)**2 + (f*x**2)**2 + (g*x)**2 + h**2
        yHat = self.a*x+self.b
        J = 0.5*sum((y-yHat)**2/sigma_2)/x.shape[0] + 0.5*sum(np.log(sigma_2))/x.shape[0]+(self.lambd/2)*(sum(self.getParams()**2))
        return J
        
    def costFunctionPrime(self, x, y):
        #Compute derivative with respect to a, b, c, d, e, f, g, h for a given x and y:
        c=Heavy_tail_regression5.c
        d=Heavy_tail_regression5.d
        e=Heavy_tail_regression5.e
        f=Heavy_tail_regression5.f
        g=Heavy_tail_regression5.g
        h=Heavy_tail_regression5.h
        sigma_2 = (c*x**5)**2 + (d*x**4)**2 +(e*x**3)**2 + (f*x**2)**2 + (g*x)**2 + h**2
        yHat = self.a*x+self.b
        dJda = sum(-x*(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.a
        dJdb = sum(-(y-yHat)/sigma_2)/x.shape[0] + self.lambd*self.b
        dJdc = sum(c*x**10/sigma_2)/x.shape[0]-sum(c*x**10*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*c        
        dJdd = sum(d*x**8/sigma_2)/x.shape[0]-sum(d*x**8*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*d        
        dJde = sum(e*x**6/sigma_2)/x.shape[0]-sum(e*x**6*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*e        
        dJdf = sum(f*x**4/sigma_2)/x.shape[0]-sum(f*x**4*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*f
        dJdg = sum(g*x**2/sigma_2)/x.shape[0]-sum(g*x**2*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*g
        dJdh = sum(h/sigma_2)/x.shape[0]-sum(h*(y-yHat)**2/sigma_2**2)/x.shape[0] + self.lambd*h       
        
        return dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg, dJdh

    def computeGradients(self, x, y, L=0):       
        dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg, dJdh = self.costFunctionPrime(x, y)
        if L==1:
            return np.asarray((dJdc, dJdd, dJde, dJdf, dJdg, dJdh))
        elif L==2:
            return np.asarray((dJda, dJdb))
        else:
            return np.asarray((dJda, dJdb, dJdc, dJdd, dJde, dJdf, dJdg, dJdh))
    
    def getParams(self, L=0):
        #Get a, b, c, d :
        if L==1:
            params = np.asarray((Heavy_tail_regression5.c, Heavy_tail_regression5.d, Heavy_tail_regression5.e, \
                                 Heavy_tail_regression5.f, Heavy_tail_regression5.g, Heavy_tail_regression5.h))
        elif L==2:
            params = np.asarray((self.a, self.b))
        else:
            params = np.asarray((self.a, self.b, Heavy_tail_regression5.c, Heavy_tail_regression5.d, Heavy_tail_regression5.e, \
                                 Heavy_tail_regression5.f, Heavy_tail_regression5.g, Heavy_tail_regression5.h))
        return params
    
    def setParams(self, params, L=0):
        #Set a, b, c, d, e, f, g, h :
        if L==1:
            Heavy_tail_regression5.c = params[0]
            Heavy_tail_regression5.d = params[1]
            Heavy_tail_regression5.e = params[2]
            Heavy_tail_regression5.f = params[3]
            Heavy_tail_regression5.g = params[4]
            Heavy_tail_regression5.h = params[5]
        elif L==2:
            self.a = params[0]
            self.b = params[1]
        else:
            self.a = params[0]
            self.b = params[1]
            Heavy_tail_regression5.c = params[2]
            Heavy_tail_regression5.d = params[3]
            Heavy_tail_regression5.e = params[4]
            Heavy_tail_regression5.f = params[5]
            Heavy_tail_regression5.g = params[6]
            Heavy_tail_regression5.h = params[7]
            
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
        c=abs(Heavy_tail_regression5.c)
        d=abs(Heavy_tail_regression5.d)
        e=abs(Heavy_tail_regression5.e)
        f=abs(Heavy_tail_regression5.f)
        g=abs(Heavy_tail_regression5.g)
        h=abs(Heavy_tail_regression5.h)

        np.random.seed()
        x=np.random.uniform(x1,x2,N)
        c_randn=np.random.normal(0,c,N)
        d_randn=np.random.normal(0,d,N)
        e_randn=np.random.normal(0,e,N)
        f_randn=np.random.normal(0,f,N)
        g_randn=np.random.normal(0,g,N)
        h_randn=np.random.normal(0,h,N)
        y = self.a*x + self.b + c_randn*x**5 + d_randn*x**4 + e_randn*x**3 + f_randn*x**2 + g_randn*x +h_randn        
        
        return np.append(x, y).reshape(2,N).transpose()
                            
                            
    