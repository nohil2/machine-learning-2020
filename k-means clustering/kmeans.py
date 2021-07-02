'''
Noah Hill
Assignment 4
'''

import numpy as np
import random
import math
import matplotlib.pyplot as plot

#run main to do both experiments
def main():
    kmeans(10)
    kmeans(30)

#kmeans algorithm
def kmeans(k):
    training_data = np.genfromtxt("optdigits.train", delimiter=",", dtype=float)
    test_data = np.genfromtxt("optdigits.test", delimiter=",", dtype=float)
    data_size = np.shape(training_data)[0]
    test_size = np.shape(test_data)[0]
    
    cluster_results = []
    cluster_amse = []
    cluster_centers = []
    
    for x in range(0, 5):
        #Generate random centers from the training data
        prev_centers = []
        centers = []
        for x in range(0, k):
            c = random.randint(0, data_size - 1)
            centers.append(training_data[c][:-1])
        clusters = []
        
        #k-means 
        same_centers = False
        while same_centers == False:
            clusters = []
            for x in range(0, k):
                clusters.append([])
            #Calculate distances and assign to clusters
            for x in range(data_size):
                mindist = 100000
                cluster = 0
                for y in range(k):
                    dist = np.sum((np.subtract(training_data[x][:-1], centers[y]))**2)
                    if dist < mindist:
                        mindist = dist
                        cluster = y
                    elif dist == mindist:
                        rand = random.randint(0, 1)
                        if rand == 0:
                            cluster = y
                clusters[cluster].append(x)
            
            #Check if centers have moved
            prev_centers = centers[:]
            centers = []
            for cluster in clusters:
                c = np.zeros_like(training_data[0][:-1])
                for x in cluster:
                    c = np.sum([c, training_data[x][:-1]], axis=0)
                c = c / np.size(cluster)
                centers.append(c)
            if np.array_equal(prev_centers, centers):
                same_centers = True
    
        cluster_results.append(clusters)
        cluster_centers.append(centers)
        
        #calculate average mean square error for the run
        amse = 0
        for centroid in range(len(centers)):
            total_dist = 0
            for item in clusters[centroid]:
                total_dist += np.sum(np.subtract(training_data[item][:-1], centers[centroid])**2)
            amse += (total_dist / np.size(clusters[centroid]))
        cluster_amse.append(amse / k)
    
    #Choose best run based on lowest average mean square error
    best_run = np.argmin(cluster_amse)
    
    #Calculate mean square separation
    mss = 0
    for a in cluster_centers[best_run]:
        for b in cluster_centers[best_run]:
            s = np.sum((np.subtract(a, b))**2)
            mss += s
    mss /= (k * (k - 1))
    
    #Calculate cluster entropies
    entropies = []
    cluster_labels = []
    for cluster in cluster_results[best_run]:
        entropy = 0
        label_count = np.zeros(k, dtype=int)
        for x in cluster:
            label = training_data[x][-1]
            label_count[int(label)] += 1
        mfl = np.argmax(label_count)
        cluster_labels.append(mfl)
        
        for x in label_count:
            if x > 0:
                entropy -= ((x / np.size(cluster)) * math.log2((x / np.size(cluster))))
        entropies.append(entropy)
    
    #Calculate mean entropy
    mean_entropy = 0
    for x in range(len(entropies)):
        mean_entropy += (entropies[x] * (np.size(cluster_results[best_run][x]) / data_size))
    
    print("AMSE: ", cluster_amse[best_run])
    print("MSS: ", mss)
    print("Mean entropy: ", mean_entropy)
    
    #Assign test data to clusters
    test_clusters = []
    for x in range(0, k):
            test_clusters.append([])
    for x in range(test_size):
        mindist = 100000
        cluster = 0
        for y in range(len(cluster_centers[best_run])):
            dist = np.sum((np.subtract(test_data[x][:-1], cluster_centers[best_run][y]))**2)
            if dist < mindist:
                mindist = dist
                cluster = y
            if dist == mindist:
                rand = random.randint(0, 1)
                if rand == 0:
                    cluster = y
        test_clusters[cluster].append(x)
    
    #calculating accuracy on test data and generating confustion matrix
    num_correct = 0
    confusion_matrix = np.zeros((10,10), dtype=int)
    for x in range(k):
        for y in test_clusters[x]:
            if cluster_labels[x] == test_data[y][-1]:
                num_correct += 1
            confusion_matrix[int(test_data[y][-1])][cluster_labels[x]] += 1
    print("Accuracy: ", ((num_correct / test_size) * 100), " %")
    print(confusion_matrix)
    
    #visualizing clusters
    for x in range(k):
        print(cluster_labels[x])
        plot.show(plot.imshow(np.reshape(cluster_centers[best_run][x], (8,8)), cmap="gray"))
