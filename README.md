# -Handwritten-Digit-Image-Classifier-using-Neural-Networks
Developed a neural network for binary classification between handwritten digits 0 and 8 from the MNIST dataset.
The neural network architecture consisted of a 784-input layer, a 300-hidden layer, and a 1-output layer, using ReLU and Sigmoid activation functions.
Binary Cross-Entropy loss function and Stochastic Gradient Descent with ADAM optimization were employed.
The model was trained on a dataset of 10,000 images and achieved high accuracy when evaluated on a separate test set of 1,000 images.

This project was developed for the Machine Learning Course (University of Patras).
First to run load_mnist.py:
1) Download the 't10k-images.idx3-ubyte','t10k-labels.idx1-ubyte','train-images.idx3-ubyte,'train-labels.idx1-ubyte'.
2) Put the files and script into the same file and change the adress where is needed.
3) By running you save only the data and labels for 0s and 8s( you change this) into csv files.

Afterwards to run Main.py change adress where is needed.
Description:
1) load_mnist.py: Load the MNIST data files and saves to csv.
2) Load_CSV_file.py: Load Images from CSV for trainning and testing.
3) ML_fun.py: It contains Sigmoid, ReLu and their derivatives.
4) Cross_Entropy.py: It contains the Cross Entropy Loss function and the derivative.
5) NN_Test.py: Use the trainned parameters to find the percent error rate of the Network.
6) Main.py: Initialize and Train the Neural Network.
