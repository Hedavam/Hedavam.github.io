import numpy as np

class LinearRegression():
    
    def __init__(self, w, score_history):
        self.w = None #going to be our w_ which stores (w, -b)
        self.score_history = [] #to store accuracy history updated each iteration
        

    def fit_analytic(self, X, y):
        '''
        This function calculates the parameter vector that minimizes squared error by solving for it explicitly by setting the gradient to 0
        
        param X: feature matrix w/ n samples
        param y: target vector w/ actual #s corresponding to each sample

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        #refer to https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/regression.html

        X_ = self.pad(X)
        XT = np.transpose(X_)
        self.w = np.linalg.inv(XT@X_)@(XT@y) 
        
    def fit_gradient(self, X, y, alpha, max_epochs):
        '''
        This function calculates the parameter vector that minimizes squared error with gradient descent algorithm
        
        param X: feature matrix w/ n samples
        param y: target vector w/ actual #s corresponding to each sample

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        
        X_ = self.pad(X)
        n_samples = X_.shape[0]
        XT = np.transpose(X_)
        self.w = np.random.rand(X_.shape[1]) * 0 
        P = XT@X_
        q = XT@y
        
        grad = (2*((P@self.w)-q))/n_samples #normalized by dividing by the number of samples to keep the gradient from being too high causing us to use a really small learning rate;  #direction is preserved, magnitude is more manageable
        
        self.score_history = [] #reset when refitting

        #loop through vector, updating vector of weights max_steps times; stopping prematurely if the algo. converges (no significant change in loss; could also check if gradient reaches (is close to) 0 - not implemented here)
        for i in range(max_epochs): 
            
            self.w -= alpha*grad #update parameters w/ gradient step
            
            #need to update the gradient inside fit; after every step down the hill, you need to recalculate and find the next best step
            grad = (2*((P@self.w)-q))/n_samples
        

        
            #different set of predictions and different accuracy every iteration because of the weights and bias changing    
            #update score history every iteration (implicitly updates predictions since score func. calls predict func.)  
            self.score_history.append(self.score(X,y))
   
            #convergence check
            if np.allclose(grad, np.zeros(len(grad))):   
                print("Gradient approach conveged")
                break
    
    def predict(self, X): #linear predictions
        '''        
        param X: feature matrix w/ n samples
        
        Returns vector of linear predictions
        '''
        
        X_ = self.pad(X)
        return X_@self.w
    
    def score(self, X, y): #coefficient of determination
        '''
        This function predicts the model's coefficient of determination (accuracy measure) representing how close our predictions are to the actual values. 0-1, with negative numbers 

        
        param X: feature matrix w/ n samples
        param y: target vector w/ actual #s corresponding to each sample

        returns a number no larger than 1 (representing perfect predictive accuracy). May return negative balues for bad models.
        '''
        #refer to coefficient of determination formula https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/regression.html

        X_ = self.pad(X)
        y_hat = self.predict(X)
        squared_error = ((y_hat-y)**2).sum()
        y_bar = y.mean()
        denom = ((y_bar-y)**2).sum()
        coef_determination = 1 - (squared_error/denom)
        return coef_determination
       
        
    def pad(self,X):
        '''
        This function helps modify our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating

        param X: matrix w/ n samples

        returns an updated feature matrix with a constant feature
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)