import numpy as np

class Perceptron:
    
    def __init__(self, w, history):
        self.w = None #going to be our w_ which stores (w, -b)
        self.history = [] #to store accuracy history updated each iteration
        
        
    def fit(self, X, y, max_steps):  #passing in Matrix X w/ n observations and n features #Passing vector y of labels w/ 0s and 1s
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        
        self.w = np.random.rand(X_.shape[1]) #Initializing random vector representing w_
      
        n = X.shape[0] #Gives us the n, the number of samples (data points)
        
        i = np.random.randint(n) #Generate a random index i between 0 and n-1
        
        yi = y[i] #Extract ith row of y
        
        xi = X_[i] #Extract the ith row of X_
        
 

        for i in range(max_steps): #loop throuh vector of weights; stopping prematurely if the algo. converges (accuracy = 1)
            
            #Perceptron update rule #refer to https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-perceptron.html
            #convert actual labels (y w/ 0 and 1 values) to -1 and 1, which helps numpy go fast with 2*label - 1
            self.w = self.w + (1*(((2*yi-1) * self.w@xi) < 0) * ((2*yi-1) *xi)) #will only add the righthandside and update self.w if labels are different 
            
            #different set of predictions and different accuracy every iteration because of the weights and bias changing    
            #update accuracy history and predictions every iteration
            self.predict(X)
            self.score(X,y)
            
            if(self.history[i] == 1.0):
                break
          
        
           
            #Update and pick a new random sample and its corresponding predictions
            i = np.random.randint(n) #Generate a random index i between 0 and n-1
            yi = y[i]
            xi = X_[i]  
            
                              
        
    def predict(self, X): #loop through all of the x features and y labels in X???? b
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) #Modifiying our given feature array by adding column of 1's so we could (in combination w/ using w_) disregard bias updating
        
        #refer to https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-perceptron.html

        return (1*((X_@self.w) >= 0)) #int(boolean) returns 0 for false and 1 for true; #@ does dot and matrix multiplication (vectorizes for speed); instead of putting boolean check inside int(), we could multiply by 1 to convert the boolean results to 0s and 1s
        
        #Returns np.array of y_ (predicted labels)
        
    def score(self, X,y): #accuracy
        accuracy = ((self.predict(X) == y).mean()) #predict(X) returns array of y_ (predicted labels); this line calculates accuracy as #y_ == y checks if prediction = label and returns 0 or 1 for each check for all samples; .mean() helps us do it all in one line without loops or writing out math
        self.history.append(accuracy)   
        return accuracy #at each iteration; #formula for accuracy is 1/n * sum (for all n samples) of the indicator of (y_ == y); basically checks for all samples if the predicted label = actual label; if so add 1 to sum, otherwise add 0; divide by # samples 
    
        
            
            
              
                
           