import numpy as np

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