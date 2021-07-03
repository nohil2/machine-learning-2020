#Noah Hill

import numpy as np

#main runs the perceptron learning function 3 times with different learning rates
#in total takes 30-40 minutes to complete all three calls
def main():
    perceptron_learning(.1)
    perceptron_learning(.01)
    perceptron_learning(.001)


#perceptron_learning trains the perceptrons and tests their accuracy
#also generates a confusion matrix on the 50th epoch
#10-13 minutes to run
def perceptron_learning(lr):
    #initialize storage for correct predictions per epoch
    epoch_test_accuracy = []
    epoch_training_accuracy = []
    
    #set up training and test data
    training_data = np.genfromtxt("mnist_train.csv", delimiter=",", dtype=float)
    training_data = training_data*(1/255) #scale values to between 0 and 1
    training_data = np.insert(training_data, 1, 1 ,axis=1) #insert bias value into data, 'x0'

    test_data = np.genfromtxt("mnist_test.csv", delimiter=",", dtype=float)
    test_data = test_data*(1/255) #scale values to between 0 and 1
    test_data = np.insert(test_data, 1, 1 ,axis=1) #insert bias value into data, 'x0'

    
    #set up perceptrons with random weights
    perceptrons = (np.random.rand(10, 785) * .1) - 0.05
    
    #train perceptrons for 50 epochs and run an epoch zero with no training
    for epoch in range(51):
        test_accuracy_count = 0  #set the number of correct predictions to zero
        training_accuracy_count = 0
        np.random.shuffle(training_data) #shuffle data each epoch
        
        if epoch > 0: #only train on epochs past epoch 0
            for row in range(len(training_data)): #for each data entry
                output = np.dot(perceptrons, training_data[row][1:]) #calculate the dot product of the perceptron weights and the entry
                target = int(training_data[row][0]*255) #determine the intended result 
             
                #when training the perceptrons:
                #change output vector to 1s and 0s to get yk
                yk = np.copy(output)
                yk[np.where(yk < 0)] = 0 #0s if <0
                yk[np.where(yk > 0)] = 1 #1s if >0
    
                #create a target vector, tk
                tk = np.zeros_like(output)
                tk[target] = 1 #with only the correct result as a 1, the rest zeros
                
                temp = lr * (tk - yk) #calculate part of the change in weights as a vector
                deltaW = np.zeros_like(perceptrons) #create an empty deltaW array, same size as perceptrons array
                for x in range(10):
                    deltaW[x] = temp[x] * training_data[row][1:] #calculate deltaWs
                    
                perceptrons = np.add(perceptrons, deltaW) #W <- deltaW + W
                #end loop      
                
        #check accuracy of perceptrons after each epoch
        for row in range(len(training_data)):
            output = np.dot(perceptrons, training_data[row][1:]) 
            target = int(training_data[row][0]*255) 
            prediction = np.argmax(output) #determine the prediction by taking the largest value from the dot product
        
            if prediction == target:
                training_accuracy_count += 1 #increase number of correct predictions if prediction is correct
                
        for row in range(len(test_data)):
            output = np.dot(perceptrons, test_data[row][1:]) 
            target = int(test_data[row][0]*255) 
            prediction = np.argmax(output) #determine the prediction by taking the largest value from the dot product
        
            if prediction == target:
                test_accuracy_count += 1 #increase number of correct predictions if prediction is correct            
        
        #track accuracy for plots as percentage
        epoch_test_accuracy.append(test_accuracy_count / 100)
        epoch_training_accuracy.append(training_accuracy_count / 600)
        
    #generate a confusion matrix for the test data set
    confusion_matrix = np.zeros((10,10), dtype=np.int32)
    for row in range(len(test_data)):
        output = np.dot(perceptrons, test_data[row][1:]) 
        target = int(test_data[row][0]*255) 
        prediction = np.argmax(output) #determine the prediction by taking the largest value from the dot product
        confusion_matrix[target][prediction] += 1
    
    #print accuracies
    print("Learning rate is: ", lr)
    print("Training set accuracy per epoch:")
    print(epoch_training_accuracy)
    print("Test set accuracy per epoch:")
    print(epoch_test_accuracy)
    print("Confusion matrix:")
    print(confusion_matrix)
