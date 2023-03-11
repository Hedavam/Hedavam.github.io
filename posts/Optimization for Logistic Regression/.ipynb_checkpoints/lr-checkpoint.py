import numpy as np

class LogisticRegression():
    
    def __init__(self, w, loss_history, score_history):
        self.w = None #going to be our w_ which stores (w, -b)
        self.loss_history = [] #to store loss history updated each iteration
        self.score_history = [] #to store accuracy history updated each iteration
        

    def gradient(self, X, y):  
        '''
        This function calculates gradient gradient of the empirical risk for logistic regression for whole training dataset 

        param X: matrix w/ n samples
        param y: vector of actual lables (0s and 1s)

        returns the gradient for the whole dataset 
        '''
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        
        sum = 0 #initialize running total of gradients accounting for each sample's gradient
        
        #loop through and do gradient formula; refer to https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html#gradient-descent
        for i in range(X_.shape[0]):
            sum += (self.sigmoid(self.w@X_[i])-y[i])*X_[i] #in each iteration we calculate the gradient for a sample, we do this for all samples and use the average for gradient descnet
        
    
        return (sum/X_.shape[0])  
        #find way to vectorize this ahahaha
        #return ((self.sigmoid(xi@self.w)-yi)*xi).mean()??
    

    def fit(self, X, y, alpha, max_epochs): 
        '''
        This function fits our logistic regression model optimized w/ batch gradient descent

        param X: matrix w/ n samples
        param y: vector of actual lables (0s and 1s)
        param alpha: the learning rate; adjusts the speed of gradient descent; higher values mean the descent will occur faster - though model may not always converge; mathematically gradient descent will converge for a small enough learning rate
        param max_epochs: specifies the number of maximum epochs (iterations) for which we will perfom the logistic regression update rule w/ gradient descent

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        
        prev_loss = np.inf #handy way to start off the loss 
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
       
        self.w = np.random.rand(X_.shape[1]) #Initializing random vector representing w_ (param vector)
        
        grad_fit = self.gradient(X,y) #calculates the gradient for whole training dataset

        # main loop
        for i in range(max_epochs): #loop through vector, updating vector of weights max_steps times; stopping prematurely if the algo. converges (no significant change in loss; could also check if gradient reaches (is close to) 0 - not implemented here)
            self.w -= alpha*grad_fit #update parameters w/ gradient step
            
            new_loss = self.empirical_risk(X, y, self.logistic_loss) #calculate new loss w/ empirical risk

            self.loss_history.append(new_loss) #add loss to loss history array

            #check if loss hasn't changed (algo. converges) and terminate if so
            if np.isclose(new_loss, prev_loss):     
                print("Conveged")
                break
            else:
                prev_loss = new_loss #update prev_loss to check vs. new_loss in next iteration 

                #different set of predictions and different accuracy every iteration because of the weights and bias changing    
                #update accuracy history and predictions every iteration
                self.predict(X)
                self.score(X,y)
                
        
    def fit_stochastic(self, X, y, alpha, max_epochs, batch_size): #add momentum param
        '''
        This function fits our logistic regression model optimized w/ stochastic gradient descent

        param X: matrix w/ n samples
        param y: vector of actual lables (0s and 1s)
        param alpha: the learning rate; adjusts the speed of gradient descent; higher values mean the descent will occur faster - though model may not always converge; mathematically gradient descent will converge for a small enough learning rate
        param max_epochs: specifies the number of maximum epochs (iterations) for which we will perfom the logistic regression update rule w/ stochastic gradient descent, after one epoch, we will go through all batches (all samples of training data)
        param batch_size: size of the random subsets of our training data, which we will use to calculate gradient

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        n = X.shape[0] #gives us number of samples
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        
        #momentum_beta = 0
        
        #if(momentum == true):
         #   momentum_beta = 0.8
            
        convergeCheck = False

        self.w = np.random.rand(X_.shape[1]) #Initializing random vector representing w_

        prev_loss = np.inf #handy way to start off the loss 
             
        #creating batches 
        for j in np.arange(max_epochs):

            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1): 
                x_batch = X[batch,:] 
                y_batch = y[batch] 
                grad_stoch = (self.gradient(x_batch, y_batch)) #calculates gradient for one batch; 


                #do gradient descent with the gradient for this batch
                self.w -= alpha*grad_stoch #update parameters w/ gradient step  #momentum on right side # gradient step not working    #weights should be indexed for momentum implementation???

   

                new_loss = self.empirical_risk(X, y, self.logistic_loss) #compute loss w/ empirical risk
               
                # check if loss hasn't changed (algo. convergences) and terminate if so
                if np.isclose(new_loss, prev_loss): 
                    convergeCheck = True
                    break

                else:
                    prev_loss = new_loss
                    self.predict(X)
                    self.score(X,y)
            
            if(convergeCheck == False):
                 self.loss_history.append(new_loss) #update loss history at the end of each epoch for sake of comparing perfomance with batch gradient descent, which also updates loss history at end of epoch
               
           

        if(convergeCheck == True):
             print("Converged")
                    
            
    def predict(self, X): #for predicted labels
        '''
        This function predicts labels for each sample according to our perceptron model; positive activation returns 1 while negative activation returns 0
        
        param X: matrix w/ n samples
        
        returns vector of predicted labels {0,1}
        '''
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        return (1*((X_@self.w) >= 0))  #int(boolean) returns 0 for false and 1 for true; #@ does dot and matrix multiplication (vectorizes for speed); instead of putting boolean check inside int(), we could multiply by 1 to convert the boolean results to 0s and 1s
        #Returns np.array of y_ (predicted labels)
    
    def score(self, X, y): 
        '''
        This function predicts the model's accuracy representing the proportion of how many predicted labels match the actual labels

        param X: matrix w/ n samples

        param y: vector of actual lables (0s and 1s)

        returns the accuracy of the perceptron as a number between 0 and 1, with 1 corresponding to perfect classification.
        '''
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        accuracy = ((self.predict(X) == y).mean()) #predict(X) returns array of y_ (predicted labels: 0 or 1 form); this line calculates accuracy as #y_ == y checks if prediction = label and returns 0 or 1 for each check for all samples; .mean() helps us do it all in one line without loops or writing out math
        self.score_history.append(accuracy)
        return accuracy

    
    
    
    def sigmoid(self,z):
        '''
        This function calculates the gradient for the whole dataset

        param X: matrix w/ n samples
        param y: vector of actual lables (0s and 1s)

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        return 1 / (1 + np.exp(-z)) #formula for logistic sigmoid; necessary for logistic loss; refer to https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#thm-gradient-descent-convergence

    def logistic_loss(self,y_hat, y): 
        '''
        This function calculates the gradient for the whole dataset

        param X: matrix w/ n samples
        param y: vector of actual lables (0s and 1s)

        no return value, but parameter vector is finalized to a good linear predictor as determined by our model 
        '''
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat)) #formula for logistic loss, which is convex, which helps us find the minimum for empirical risk, refer to https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#thm-gradient-descent-convergence

    def empirical_risk(self, X, y, loss): 
        '''
        This function calculates the gradient for the whole dataset

        param X: matrix w/ n samples
        param y: vector of actual labels (0s and 1s)
        param loss: loss function we choose for our empirical risk minimization, in this case we chose logistic loss        
        '''
        X_ = self.pad(X) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        activation = X_@self.w #predicted weights; linear predictions
        return loss(activation, y).mean()  #formula for empirical risk, which we are trying to minimize, refer to https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#thm-gradient-descent-convergence


    def pad(self,X):
        '''
        This function helps modify our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating

        param X: matrix w/ n samples

        returns an updated feature matrix with a constant feature
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)