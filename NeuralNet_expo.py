#----------------------- ConstantineGk ------------------------------------#
import numpy as np
import math
import matplotlib.pyplot as plt

#--------------------------Load training data-----------------------------#
Train_data = np.loadtxt('Train_MNIST.txt', dtype = 'float', delimiter = ',');

# Predict function using weights of trainned NN
def predict(Z,A1,B1,A2,B2):
    Z = np.reshape(Z,(784,1))
    W1 = np.dot(A1, Z) + B1
    Z1 = relu(W1)
    W2 = np.dot(A2, Z1) + B2
    y = sigmoid(W2)
    return y 

# Sigmoid
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

# Sigmoid derivative
def sigmoid_der(x):
    return np.exp(-x) / ( (1 + np.exp(-x)) ** 2 )

# Hidden layer activation (ReLu)
def relu(x):
    return (np.maximum(0,x))

# ReLU derivative
def relu_der_(x):
    if x < 0: y = 0
    elif x >= 0: y = 1
    return y
relu_der = np.vectorize(relu_der_)

# Loss function Cross Entropy
def cross_Entropy(y_real, y_hat):
    y_hat = np.squeeze(y_hat)
    if y_real == 0:
        return -np.log(y_hat + (10**-100) )
    else:
        return -np.log(1 - y_hat + (10**-100) )

# Cross-Entropy derivative
def cross_E_grad(y_real, y_hat):
    y_hat = np.squeeze(y_hat)
    if y_real == 0:
        return - 1/(y_hat + (10**-100) )
    else:
        return 1/(1 - y_hat + (10**-100) )

#-------------------------------------------------------------#
# Initialize the weights and biases for the first and second layers
A1 = np.random.normal(loc=0, scale=1/(300+784), size=(300, 784))
A2 = np.random.normal(loc=0, scale=1/(1+300), size=(1, 300))
B1 = np.zeros((300,1))
B2 = np.zeros(1).reshape(1,1)

# Initialize loss matrix
loss_matrix = []

# Choose a learning rate
learning_rate = 0.01

# Set the number of epochs
epochs = 5

# ADAM parameters
lamda = 0.1
c = 0.01

# Train the neural network
for i in range(epochs):
    np.random.shuffle(Train_data)
    for iteration in range(len(Train_data)):
        y_real = Train_data[iteration][-1]
        Z = Train_data[iteration][:-1]
        Z = np.reshape(Z,(784,1))
        
        # Forward pass
        W1 = np.dot(A1, Z) + B1
        Z1 = relu(W1)
        W2 = np.dot(A2, Z1) + B2
        y_hat = sigmoid(W2)

        # LOSS
        loss_matrix.append(cross_Entropy(y_real, y_hat))

        # Gradients
        U2 = cross_E_grad(y_real, y_hat)
        V2 = np.multiply(U2 , sigmoid_der(W2))
        U1 = np.dot( A2.T, V2)
        V1 = np.multiply(U1 , relu_der(W1))

        A2_grad = np.dot(V2, Z1.T)
        B2_grad = V2
        A1_grad = np.dot(V1,Z.T)
        B1_grad = V1

        # ADAM
        if iteration == 0:
            P_A2 = A2_grad**2
            P_B2 = B2_grad**2
            P_A1 = A1_grad**2
            P_B1 = B1_grad**2
        else:
            P_A2 = (1 - lamda) * P_A2 + A2_grad**2
            P_B2 = (1 - lamda) * P_B2 + B2_grad**2
            P_A1 = (1 - lamda) * P_A1 + A1_grad**2
            P_B1 = (1 - lamda) * P_B1 + B1_grad**2
        
        # Update weights and biases
        A2 -= learning_rate * A2_grad / np.sqrt(c + P_A2) 
        B2 -= learning_rate * B2_grad / np.sqrt(c + P_B2) 
        A1 -= learning_rate * A1_grad / np.sqrt(c + P_A1) 
        B1 -= learning_rate * B1_grad / np.sqrt(c + P_B1) 
        

# Normilize Loss for plotting
lista = []
for i in range(0,len(loss_matrix),400):
    data = sum(loss_matrix[i:i + 400])/400
    lista.append(data)
plt.plot(lista)
plt.show()
#--------------------------#
print('Trained')

# Load Final Data
Test_data = np.loadtxt('Test_MNIST.txt', dtype = 'float', delimiter = ',');

# Calculate percent error 
N_mist_8=0
N_mist_0=0
for i in range(0,len(Test_data)):
    y = predict(Test_data[i][:-1],A1,B1,A2,B2)
    if y<=0.5 and Test_data[i][-1] == 0:N_mist_0 += 1
    if y>0.5 and Test_data[i][-1] == 8:N_mist_8 += 1

error_p = 0.5 * ( N_mist_0 / len(Test_data)) + 0.5 * ( N_mist_8 / len(Test_data))

print( (N_mist_0/len(Test_data)) , (N_mist_8/len(Test_data)) , error_p )


