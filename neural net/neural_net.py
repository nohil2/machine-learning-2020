#Noah Hill

import numpy as np

#main calls the neural_net function multiple times
#to fulfill all the experiment requirements
def main():
    neural_net(20, 0)
    neural_net(50, 0)
    neural_net(100, 0)
    neural_net(100, 1)
    neural_net(100, 2)

#the sigmoid activation function
def sigmoid(x):
    return (1.0 / (1 + np.exp(-x)))

#function to quickly do part of hidden node error calculation
def hid_error(x):
    return x * (1 - x)

#trains a neural net with a provided number of hidden nodes; mode indicates data set size
def neural_net(nodes, mode):
    #set learning rate
    lr = 0.1
    
    #vectorize sigmoid and hid_error for better numpy use
    v_sig = np.vectorize(sigmoid)
    v_hid_error = np.vectorize(hid_error)
    
    #initialize storage for correct predictions per epoch
    epoch_test_accuracy = []
    epoch_training_accuracy = []
    
    #set up training and test data
    training_data = np.genfromtxt("mnist_train.csv", delimiter=",", dtype=float)
    training_data = training_data*(1/255) #scale values to between 0 and 1
    training_data = np.insert(training_data, 1, 1 ,axis=1) #insert bias value into data, 'x0'
    #mode 0 for full data set
    #mode 1 for half data set
    data_size = len(training_data)
    
    if mode == 1:
        data_size = int(data_size / 2)
        
    #mode 2 for quarter data set    
    if mode == 2:
        data_size = int(data_size / 4)
        
    
    test_data = np.genfromtxt("mnist_test.csv", delimiter=",", dtype=float)
    test_data = test_data*(1/255) #scale values to between 0 and 1
    test_data = np.insert(test_data, 1, 1 ,axis=1) #insert bias value into data, 'x0'

    
    #set up inital random weights
    hidden_weights = (np.random.rand(nodes, 785) * .1) - 0.05
    output_weights = (np.random.rand(10, (nodes + 1)) * .1) - 0.05
    
    #epoch 0
    test_accuracy_count = 0  
    training_accuracy_count = 0
    for row in range(data_size):
        data = training_data[row] #make an alias for the entry
        label = int(data[0] * 255) #identify the label
        target = np.full((10), .1) #create target vector
        target[label] = .9
        
        #input to hidden activations
        hidden_activation = np.dot(hidden_weights, data[1:])
        hidden_activation = v_sig(hidden_activation)
        hidden_activation = np.insert(hidden_activation, 0, 1)
        
        #hidden to output activations
        output_activation = np.dot(output_weights, hidden_activation)
        output_activation = v_sig(output_activation)
        prediction = np.argmax(output_activation) #determine the prediction by taking the largest value from the output activation
    
        if prediction == label:
            training_accuracy_count += 1 #increase number of correct predictions if prediction is correct
                
    for row in range(len(test_data)):
        data = training_data[row] #make an alias for the entry
        label = int(data[0] * 255) #identify the label
        target = np.full((10), .1) #create target vector
        target[label] = .9
        
        #input to hidden activations
        hidden_activation = np.dot(hidden_weights, data[1:])
        hidden_activation = v_sig(hidden_activation)
        hidden_activation = np.insert(hidden_activation, 0, 1)
        
        #hidden to output activations
        output_activation = np.dot(output_weights, hidden_activation)
        output_activation = v_sig(output_activation)
        prediction = np.argmax(output_activation) #determine the prediction by taking the largest value from the output activation
    
        if prediction == label:
            test_accuracy_count += 1 #increase number of correct predictions if prediction is correct 
    epoch_test_accuracy.append(test_accuracy_count / 100)
    epoch_training_accuracy.append(training_accuracy_count / 600)            
    #end epoch 0
    
    
    #train neural net for 50 epochs
    for epoch in range(50):
        #set the number of correct predictions to zero
        test_accuracy_count = 0  
        training_accuracy_count = 0
        
        #shuffle data each epoch
        np.random.shuffle(training_data) 
        
        for row in range(data_size): #for each data entry
            data = training_data[row] #make an alias for the entry
            label = int(data[0] * 255) #identify the label
            target = np.full((10), .1) #create target vector
            target[label] = .9
            
            #input to hidden activations
            hidden_activation = np.dot(hidden_weights, data[1:])
            hidden_activation = v_sig(hidden_activation)
            hidden_activation = np.insert(hidden_activation, 0, 1)
            
            #hidden to output activations
            output_activation = np.dot(output_weights, hidden_activation)
            output_activation = v_sig(output_activation)
            
            #calculate output error
            temp = np.subtract(np.ones_like(output_activation), output_activation)
            temp2 = np.subtract(target, output_activation)
            output_error = np.multiply(np.multiply(output_activation, temp), temp2)      
            
            #calculate hidden error
            hidden_error = v_hid_error(hidden_activation[1:])
            hidden_error = np.multiply(hidden_error, np.dot(output_error, np.delete(output_weights, 0, 1)))
            
            #calculate deltaWs
            hidden_delta = np.multiply(lr, np.dot(np.reshape(hidden_error, (nodes, 1)), np.transpose(np.reshape(data[1:], (np.size(data[1:]), 1)))))
            output_delta = np.multiply(lr, np.dot(np.reshape(output_error, (np.size(output_error), 1)), np.transpose(np.reshape(hidden_activation, (np.size(hidden_activation), 1)))))
            
            #change weights
            hidden_weights = np.add(hidden_weights, hidden_delta)    
            output_weights = np.add(output_weights, output_delta)
                
            #end loop   
            
        #check accuracy of neural net after each epoch
        for row in range(data_size):
            data = training_data[row] #make an alias for the entry
            label = int(data[0] * 255) #identify the label
            target = np.full((10), .1) #create target vector
            target[label] = .9
            
            #input to hidden activations
            hidden_activation = np.dot(hidden_weights, data[1:])
            hidden_activation = v_sig(hidden_activation)
            hidden_activation = np.insert(hidden_activation, 0, 1)
            
            #hidden to output activations
            output_activation = np.dot(output_weights, hidden_activation)
            output_activation = v_sig(output_activation)
            prediction = np.argmax(output_activation) #determine the prediction by taking the largest value from the output activation
        
            if prediction == label:
                training_accuracy_count += 1 #increase number of correct predictions if prediction is correct
                
        for row in range(len(test_data)):
            data = training_data[row] #make an alias for the entry
            label = int(data[0] * 255) #identify the label
            target = np.full((10), .1) #create target vector
            target[label] = .9
            
            #input to hidden activations
            hidden_activation = np.dot(hidden_weights, data[1:])
            hidden_activation = v_sig(hidden_activation)
            hidden_activation = np.insert(hidden_activation, 0, 1)
            
            #hidden to output activations
            output_activation = np.dot(output_weights, hidden_activation)
            output_activation = v_sig(output_activation)
            prediction = np.argmax(output_activation) #determine the prediction by taking the largest value from the output activation
        
            if prediction == label:
                test_accuracy_count += 1 #increase number of correct predictions if prediction is correct            
        
        #track accuracy for plots as percentage
        epoch_test_accuracy.append(test_accuracy_count / 100)
        epoch_training_accuracy.append(training_accuracy_count / 600)
        
    #generate a confusion matrix for the test data set
    confusion_matrix = np.zeros((10,10), dtype=np.int32)
    for row in range(len(test_data)):
        data = training_data[row] #make an alias for the entry
        label = int(data[0] * 255) #identify the label
        target = np.full((10), .1) #create target vector
        target[label] = .9
        
        #input to hidden activations
        hidden_activation = np.dot(hidden_weights, data[1:])
        hidden_activation = v_sig(hidden_activation)
        hidden_activation = np.insert(hidden_activation, 0, 1)
        
        #hidden to output activations
        output_activation = np.dot(output_weights, hidden_activation)
        output_activation = v_sig(output_activation)
        prediction = np.argmax(output_activation) #determine the prediction by taking the largest value from the output activation
        confusion_matrix[label][prediction] += 1
    
    #print accuracies
    print("Learning rate is: ", lr)
    print("Mode is: ", mode)
    print("Training set accuracy per epoch:")
    print(epoch_training_accuracy)
    print("Test set accuracy per epoch:")
    print(epoch_test_accuracy)
    print("Confusion matrix:")
    print(confusion_matrix)
