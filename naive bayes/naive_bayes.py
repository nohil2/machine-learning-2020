'''
Noah Hill
Assignment 3
'''

import numpy as np
import sys

#command line arguments should be training data file then test data file
train = sys.argv[1]
test = sys.argv[2]
#train = "yeast_training.txt"
#test = "yeast_test.txt"

training_data = np.genfromtxt(train, dtype=float)
test_data = np.genfromtxt(test, dtype=float)
data_shape = np.shape(training_data)
num_attributes = data_shape[1] - 1
test_cases = np.shape(test_data)[0]

#find classes
classes = []
for row in training_data:
    classes.append(int(row[-1]))
copy = np.copy(classes)
classes = np.unique(classes)
class_counts = []
for c in classes:
    class_counts.append(np.count_nonzero(copy == c))

class_probabilities = np.divide(class_counts, np.size(copy))

#mu and sig storage
class_mu = []
class_sig = []

#training
for classifier in classes:
    pos_instance = []
    mus = []
    sigs = []
    for x in range(num_attributes):
        pos_instance.append([])
  
    for x in range(num_attributes):
        for row in training_data:
            if row[-1] == classifier:
                pos_instance[x].append(row[x])      
        mu = (np.sum(pos_instance[x]) / np.size(pos_instance[x]))
        sig = (np.sum((pos_instance[x] - mu)**2) / np.size(pos_instance[x]))**0.5
        if sig < 0.01:
            sig = 0.01
        print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (classifier, x+1, mu, sig))
        mus.append(mu)
        sigs.append(sig)
        
    class_mu.append(mus)
    class_sig.append(sigs)


#classifying test data
accuracy = 0
count = -1

for row in range(test_cases):
    probabilities = []
    for x in range(len(classes)):
        prob = np.log2((class_probabilities[x]))
        for y in range(num_attributes):
            a = 1 / (((2 * np.pi)**0.5) * class_sig[x][y])
            b = -((test_data[row][y] - class_mu[x][y])**2/(2 * (class_sig[x][y])**2))
            c = a * np.e**b
            if c > 0:
                prob += np.log2(c)        
        if prob < 0:
            prob = 2**prob
        probabilities.append(prob)

    #classify based on largest probability
    max_prob = np.amax(probabilities)
    test_class = np.argwhere(np.isclose(probabilities, max_prob, atol=0.0000000000000000000000000000000001))
    test_class = test_class.flatten()

    prediction = []
    for x in test_class:
        prediction.append(classes[x])
    for x in prediction:
        if x == test_data[row][-1]:
            if np.size(prediction) > 1:
                accurate = 1 / np.size(prediction)
            else: 
                accurate = 1
            break
        else:
            accurate = 0

    accuracy += accurate
    print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (row+1, prediction[0], max_prob, test_data[row][-1], accurate))
print("classification accuracy=%6.4f"%(accuracy / test_cases * 100))
    
    
