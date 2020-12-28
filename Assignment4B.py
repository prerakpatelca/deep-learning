""" This program uses multi-layer perceptrons on 5 data sets four from the Assignment 4A and the one from the UCI machine learning.
Source: https://archive.ics.uci.edu/ml/datasets/Crop+mapping+using+fused+optical-radar+data+set

Prerak Patel, Student, Mohawk College, 2020
"""
# importing libraries
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy as np
import csv

## READ IT
def read(file):
    train_data_file = open(file,"r")

    # creating CSV readers
    csv_reader = csv.reader(train_data_file, delimiter=",")

    # declaring the arrays for the storing data
    data_set = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for row in csv_reader:
        data_set += [[str(num) for num in row]]

    # converting the data set to np array for slicing
    data_set = np.array(data_set)

    # shuffling the array for each run
    np.random.shuffle(data_set)

    data_set = np.array((data_set.astype(np.float)))

    # slicing the data_set for the data
    data = data_set[:,:-1]
    # slicing the data_set for the labels
    labels = data_set[:,-1]

    # using the first 80% of the data set into training data
    train_data = data[:int(len(data_set)*0.8)]
    train_labels = labels[:int(len(data_set)*0.8)]

    # using the other 20% of the data set into testing data
    test_data = data[int(len(data_set)*0.8):]
    test_labels = labels[int(len(data_set)*0.8):]
    return train_data, train_labels, test_data, test_labels

# creating file array
files = ['000825410_1.csv', '000825410_2.csv', '000825410_3.csv', '000825410_4.csv', 'uci.csv']

# hidden layers array for the different file
layers = [(40,),(40,), (50,40,30,20, 10, 7, 5, 3),(50,40, 30, 20, 10, 7, 5, 3),(160,120,80,40,30,20,10)]

solvers = ['adam','adam','lbfgs','lbfgs','adam']

# looping through each file and run the algorithm on each file
for index in range(len(files)):
    train_data, train_labels, test_data, test_labels = read(files[index])

    print("\nFile: " + files[index])

## Decision Tree Classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_labels)
    accuracy = clf.score(test_data, test_labels)

    # printing decision tree accuracy
    print("Decision Tree: " + str(np.around(accuracy*100,decimals=1)) + "% Accuracy")

## MLP Classifier with different hidden layers and parameters
    clf = MLPClassifier(hidden_layer_sizes = layers[index],activation = 'relu', learning_rate = 'adaptive', solver=solvers[index], max_iter=5000)
    clf.fit(train_data, train_labels)
    accuracy = clf.score(test_data, test_labels)

    # printing parameters used, accuracy and number of iterations
    print("MLP: hidden layers = " + str(layers[index]) + "activation = 'relu', learning_rate = 'adaptive', solver = '" + str(solvers[index]) + "', max_iter=5000 " + str(np.around(accuracy*100,decimals=1)) + " : % Accuracy, " + str(clf.n_iter_) + " iterations")




