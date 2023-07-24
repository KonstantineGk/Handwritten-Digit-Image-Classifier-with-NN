import idx2numpy
import numpy as np
#---------------------------------------------------------------------------#
test_images = r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\t10k-images.idx3-ubyte'
test_arr = idx2numpy.convert_from_file(test_images).reshape([10000,784]) / 255

test_labels = r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\t10k-labels.idx1-ubyte'
test_lab = idx2numpy.convert_from_file(test_labels).reshape([10000,1])

test_data = np.concatenate((test_arr,test_lab), axis = 1)
temp = []
for i in range(0,len(test_data)):
    if test_data[i][-1] == 0 or test_data[i][-1] == 8:
        temp.append(test_data[i])
test_0_8 = np.array(temp)
#---------------------------------------------------------------------------#

tr_images = r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\train-images.idx3-ubyte'
tr_arr = idx2numpy.convert_from_file(tr_images).reshape([60000,784]) / 255

tr_labels = r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\train-labels.idx1-ubyte'
tr_lab = idx2numpy.convert_from_file(tr_labels).reshape([60000,1])

train_data = np.concatenate((tr_arr,tr_lab), axis = 1)
temp0 = []
for i in range(0,len(train_data)):
    if train_data[i][-1] == 0 or train_data[i][-1] == 8:
        temp0.append(train_data[i])
train_0_8 = np.array(temp0)




# Save to txt
np.savetxt(r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\Test_MNIST.csv',test_0_8, delimiter=',')
np.savetxt(r'C:\Users\Eygenia\Desktop\Costantine\Neural Network MNIST\Train_MNIST.csv',train_0_8, delimiter=',')
