import numpy as np
import math
import matplotlib.pyplot as plt
#--------------------------Load training data-----------------------------#
Train_data = np.loadtxt('Train_MNIST.txt', dtype = 'float', delimiter = ',');
#---------------------------------------------------------------------------#
# Predict function
def predict(X,W1,b,W2,b2):
    X = np.reshape(X,(1,784))
    I_H1 = np.dot(X, W1) + b1
    O_H1 = relu(I_H1)
    I_OL = np.dot(O_H1, W2) + b2
    y_hat = sigmoid(I_OL)
    return y_hat

# Sigmoid
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

# Sigmoid derivative
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hidden layer activation (ReLu)
def relu(x):
    return (np.maximum(0,x))

# ReLU derivative
def relu_der_(x):
    if x < 0: y = 0
    elif x > 0: y = 1
    return y
relu_der = np.vectorize(relu_der_)

# Loss function Cross Entropy
def cross_Entropy(y_real, y_hat):
    if y_real == 8:
        return -np.log(y_hat + 10**-100)
    else:
        return -np.log(1 - y_hat + 10**-100)

# Binary Cross-Entropy derivative
def cross_E_grad(y_real, y_hat):
    if y_real == 8:
        return - 1/(y_hat + 10**-100)
    else:
        return 1/(1 - y_hat + 10**-100)

#-------------------------------------------------------------#
# Initialize the weights and biases for the first and second layers
W1 = np.random.normal(loc=0, scale=1/(2+20), size=(784, 300))
W2 = np.random.normal(loc=0, scale=1/(1+20), size=(300, 1))
b1 = np.zeros((1,300))
b2 = np.zeros(1).reshape(1,1)
loss_matrix = [] # Initialize loss matrix
# Choose a learning rate
learning_rate = np.array([0.01]).reshape(1,1)
# Set the number of epochs
epochs = 80

# Train the neural network
for i in range(epochs):
    print(i)
    for j in range(len(Train_data)):
        # Forward pass
        y_real = Train_data[j][-1]
        X = Train_data[j][:-1]
        X = np.reshape(X,(1,784))
        I_H1 = np.dot(X, W1) + b1
        O_H1 = relu(I_H1)
        I_OL = np.dot(O_H1, W2) + b2
        y_hat = sigmoid(I_OL)

        # Gradients
        ce_grad = cross_E_grad(y_real, y_hat)      
        grad_b2 = ce_grad * sigmoid_der(I_OL)
        grad_W2 = np.dot(O_H1.T, grad_b2)
        error_grad_upto_H1 = np.dot(grad_b2 , W2.T)
        grad_b1 = error_grad_upto_H1 * relu_der(I_H1)
        grad_W1 = np.dot(X.T, grad_b1)

        # Update weights and biases
        b2 -= learning_rate * grad_b2
        W2 -= learning_rate * grad_W2
        b1 -= learning_rate * grad_b1
        W1 -= learning_rate * grad_W1
        
        # LOSS
        loss = cross_Entropy(y_real, y_hat)
        loss_matrix.append(np.squeeze(loss))
        
#------------- Normilize Loss for plotting ------------------------#
chunk_size = 400
lista = []
for i in range(0,epochs):
    data = sum(loss_matrix[i*chunk_size:i*chunk_size + chunk_size])
    lista.append(data)
plt.plot(lista)
plt.show()
#------------------------------------------------------------------#
print('Trained')
#----------------------- Load Final Data --------------------------#
Test_data = np.loadtxt('Test_MNIST.txt', dtype = 'float', delimiter = ',');
N_mist_8=0
N_mist_0=0
for i in range(0,len(Test_data)):
    y = predict(Test_data[i][:-1],W1,b1,W2,b2)
    if y<0.5 and Test_data[i][-1] == 0:N_mist_0 += 1
    if y>0.5 and Test_data[i][-1] == 8:N_mist_8 += 1
error_p = 0.5 * ( N_mist_0 / len(Test_data)/2) + 0.5 * ( N_mist_8 / len(Test_data)/2) # Calculate percent error 
print(N_mist_0 / len(Test_data)/2)
print(N_mist_8 / len(Test_data)/2)
print(error_p)

