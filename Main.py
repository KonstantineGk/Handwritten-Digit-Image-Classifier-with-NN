#----------------------- ConstantineGk ------------------------------------#
import numpy as np
import math
import matplotlib.pyplot as plt
from Load_CSV_files import LoadCSVfiles
import ML_fun as ml
import Cross_Entropy as ce
from NN_Test import Calc_error

def main():
    #--------------------------Load training data-----------------------------#
    Train_data, Test_data = LoadCSVfiles()

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
    epochs = 1

    # ADAM parameters
    lamda = 0.1
    c = 0.01

    # Train the neural network
    for i in range(epochs):
        print(i)
        np.random.shuffle(Train_data)
        for iteration in range(len(Train_data)):
            y_real = Train_data[iteration][-1]
            Z = Train_data[iteration][:-1]
            Z = np.reshape(Z,(784,1))
            
            # Forward pass
            W1 = np.dot(A1, Z) + B1
            Z1 = ml.relu(W1)
            W2 = np.dot(A2, Z1) + B2
            y_hat = ml.sigmoid(W2)

            # LOSS
            loss_matrix.append(ce.cross_Entropy(y_real, y_hat))

            # Gradients
            U2 = ce.cross_E_grad(y_real, y_hat)
            V2 = np.multiply(U2 , ml.sigmoid_der(W2))
            U1 = np.dot( A2.T, V2)
            V1 = np.multiply(U1 , ml.relu_der(W1))

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
            
    print('Trained')
    
    # Error Rate
    Calc_error(A1,A2,B1,B2,Test_data)

if __name__ == '__main__':
    main()
