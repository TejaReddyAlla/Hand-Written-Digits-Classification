import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary


    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    

    data_preprocess_train = np.zeros(shape=(50000, 784))
    data_preprocess_validate = np.zeros(shape=(10000, 784))
    data_preprocess_test = np.zeros(shape=(10000, 784))
    data_preprocess_trainLabel = np.zeros(shape=(50000,))
    data_preprocess_validationLabel = np.zeros(shape=(10000,))
    data_preprocess_tesetLabel = np.zeros(shape=(10000,))
 
    len_trainData = 0
    len_validationData = 0
    len_testData = 0
    len_trainLabel = 0
    len_validationLabel = 0
 
    for i in mat:
        if "train" in i:
            label = i[-1]   
            tup = mat.get(i)
            sap = range(tup.shape[0])
            t_permutation = np.random.permutation(sap)
            t_length = len(tup)   
            len_tag = t_length - 1000   

            data_preprocess_train[len_trainData:len_trainData + len_tag] = tup[t_permutation[1000:], :]
            len_trainData += len_tag

            data_preprocess_trainLabel[len_trainLabel:len_trainLabel + len_tag] = label
            len_trainLabel += len_tag

            data_preprocess_validate[len_validationData:len_validationData + 1000] = tup[t_permutation[0:1000], :]
            len_validationData += 1000

            data_preprocess_validationLabel[len_validationLabel:len_validationLabel + 1000] = label
            len_validationLabel += 1000

        elif "test" in i:
            label = i[-1]
            tup = mat.get(i)
            sap = range(tup.shape[0])
            t_permutation = np.random.permutation(sap)
            t_length = len(tup)
            data_preprocess_tesetLabel[len_testData:len_testData + t_length] = label
            data_preprocess_test[len_testData:len_testData + t_length] = tup[t_permutation]
            len_testData += t_length

    train_size = range(data_preprocess_train.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = data_preprocess_train[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = data_preprocess_trainLabel[train_perm]

    validation_size = range(data_preprocess_validate.shape[0])
    valid_permutation = np.random.permutation(validation_size)
    validation_data = data_preprocess_validate[valid_permutation]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = data_preprocess_validationLabel[valid_permutation]

    test_size = range(data_preprocess_test.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = data_preprocess_test[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = data_preprocess_tesetLabel[test_perm]

    # Feature selection
    # Your code here.

    #removing the redundant data
    remove = np.all(train_data == train_data[0, :], axis=0)

    train_data = train_data[:, ~remove]
    validation_data = validation_data[:, ~remove]
    test_data = test_data[:, ~remove]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""


    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    oneOfK = np.zeros((training_label.shape[0], 10))
    oneOfK[np.arange(training_label.shape[0], dtype="int"), training_label.astype(int)] = 1
    training_label = oneOfK

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # Forward Propagation
    one = np.array(training_data)
    two = np.array(np.ones(training_data.shape[0]))

    training_data = np.column_stack((one, two))


    hl = sigmoid(np.dot(training_data, w1.T))

    hl = np.column_stack((hl, np.ones(hl.shape[0])))

    ol = sigmoid(np.dot(hl, w2.T))

    delta = ol - training_label

    # Gradient decent w2
    w2_gradient = np.dot(delta.T, hl)

    # Using formula from (11 & 12)
    w1_gradient = np.dot(((1 - hl) * hl * (np.dot(delta, w2))).T, training_data)

    # Remove zero row
    w1_gradient = np.delete(w1_gradient, n_hidden, 0)

    # calculating obj_val
    n = training_data.shape[0]
    error = (np.sum(-1 * (training_label * np.log(ol) + (1 - training_label) * np.log(1 - ol)))) / n
    obj_val = error + ((lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    # Partial derivative of new objective function with respect to weight hidden to output layer
    w2_gradient = (w2_gradient + (lambdaval * w2)) / n
    # Partial derivative of new objective function with respect to weight input to hidden layer
    w1_gradient = (w1_gradient + (lambdaval * w1)) / n

    # calculating obj_grad
    obj_grad = np.array([])
    obj_grad = np.concatenate((w1_gradient.flatten(), w2_gradient.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    # adding bias node in input vector
    bias = np.ones(len(data))
    data = np.column_stack([data, bias])

    res1 = data.dot(w1.T)

    sigmoid_res_1 = sigmoid(res1)

    sigmoid_bias = np.ones(len(data))

    sigmoid_res_1 = np.column_stack([sigmoid_res_1, sigmoid_bias])

    ol = sigmoid_res_1.dot(w2.T)
    
    sigmoid2 = sigmoid(ol)
    
    labels = np.argmax(sigmoid2,axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
timer = time.time()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 60 # 50 is the default

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
# regularization parameter is initialized to 0
lambdaval = 15

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 200}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

getTime = time.time()-timer
print('\n Execution completed in ' + str(getTime)+ 'seconds')

obj = [n_hidden, w1, w2, lambdaval ]
pickle.dump(obj, open('params.pickle', 'wb'))