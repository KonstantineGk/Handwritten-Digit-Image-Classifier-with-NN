# -Handwritten-Digit-Image-Classifier-using-Neural-Networks
Developed a neural network for binary classification between handwritten digits 0 and 8 from the MNIST dataset.
The neural network architecture consisted of a 784-input layer, a 300-hidden layer, and a 1-output layer, using ReLU and Sigmoid activation functions.
Binary Cross-Entropy loss function and Stochastic Gradient Descent with ADAM optimization were employed.
The model was trained on a dataset of 10,000 images and achieved high accuracy when evaluated on a separate test set of 1,000 images.

This project was developed for the Machine Learning Course (University of Patras).
First to run load_mnist.py:
1) Download the 't10k-images.idx3-ubyte','t10k-labels.idx1-ubyte','train-images.idx3-ubyte,'train-labels.idx1-ubyte'.
2) Put the files and script into the same directory.
3) By running you save only the data and labels for 0s and 8s( you change this) into txt files.

Description:
1) Load the data from before.
2) predict: Define the front propagation of the NN to use after trainning.
3) sigmoid: Define the sigmoid function.
4) sigmoid_der: Define the sigmoid derivative.
5) relu: Define ReLu.
6) relu_der: Define the ReLu derivative. After vectorize for use with np.arrays.
7) 
