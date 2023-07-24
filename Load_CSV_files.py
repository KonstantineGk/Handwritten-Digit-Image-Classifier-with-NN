import numpy as np

def LoadCSVfiles():
    # Load Train Images
    Train_data = np.loadtxt(r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\Train_MNIST.csv', dtype = 'float', delimiter = ',')
    
    # Load Test Images
    Test_data = np.loadtxt(r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\Test_MNIST.csv', dtype = 'float', delimiter = ',')
    
    return Train_data, Test_data
