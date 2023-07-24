import numpy as np
import ML_fun as ml

# Predict function using weights of trainned NN
def predict(Z,A1,B1,A2,B2):
    Z = np.reshape(Z,(784,1))
    W1 = np.dot(A1, Z) + B1
    Z1 = ml.relu(W1)
    W2 = np.dot(A2, Z1) + B2
    y = ml.sigmoid(W2)
    return y 

def Calc_error(A1,A2,B1,B2,Test_data): 
    N_mist_8=0
    N_mist_0=0
    
    for i in range(0,len(Test_data)):
        y = predict(Test_data[i][:-1],A1,B1,A2,B2)
        if y<=0.5 and Test_data[i][-1] == 0:N_mist_0 += 1
        if y>0.5 and Test_data[i][-1] == 8:N_mist_8 += 1

    error_p = 0.5 * ( N_mist_0 / len(Test_data)) + 0.5 * ( N_mist_8 / len(Test_data))

    print( (N_mist_0/len(Test_data)) , (N_mist_8/len(Test_data)) , error_p )
    