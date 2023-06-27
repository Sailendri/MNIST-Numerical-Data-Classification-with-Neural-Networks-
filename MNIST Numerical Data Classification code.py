#import libraries and functions to load the data
from digits import get_mnist
from matplotlib import pyplot as plt
import numpy as np
import ast
import sys
import numpy.testing as npt
import pytest
import random
from IPython.core.debugger import set_trace

#Load and visualize data
random.seed(1)
np.random.seed(1)
trX, trY, tsX, tsY = get_mnist()
# We need to reshape the data everytime to match the format (d,m), where d is␣dimensions (784) and m is number of samples
trX = trX.reshape(-1, 28*28).T
trY = trY.reshape(1, -1)
tsX = tsX.reshape(-1, 28*28).T
tsY = tsY.reshape(1, -1)
# Lets examine the data and see if it is normalized
print('trX.shape: ', trX.shape)
print('trY.shape: ', trY.shape)
print('tsX.shape: ', tsX.shape)
print('tsY.shape: ', tsY.shape)
print('Train max: value = {}, Train min: value = {}'.format(np.max(trX), np.min(trX)))
print('Test max: value = {}, Test min: value = {}'.format(np.max(tsX), np.min(tsX)))
print('Unique labels in train: ', np.unique(trY))
print('Unique labels in test: ', np.unique(tsY))
# Let's visualize a few samples and their labels from the train and test␣datasets.
print('\nDisplaying a few samples')
visx = np.concatenate((trX[:,:50],tsX[:,:50]), axis=1).reshape(28,28,10,10).transpose(2,0,3,1).reshape(28*10,-1)
visy = np.concatenate((trY[:,:50],tsY[:,:50]), axis=1).reshape(10,-1)
print('labels')
print(visy)
plt.figure(figsize = (8,8))
plt.axis('off')
plt.imshow(visx, cmap='gray');

def relu(Z):
#Computes relu activation of input Z
#Inputs:
#Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n'␣dimension
#Outputs:
#A: where A = ReLU(Z) is a numpy.ndarray (n, m) representing 'm' samples␣each of 'n' dimension
#cache: a dictionary with {"Z", Z}
    cache = {}
    A = np.maximum(0,Z)
    cache={"Z":Z}
    return A, cache

def relu_der(dA, cache):

#Computes derivative of relu activation
#Inputs:
#dA: derivative from the subsequent layer of dimension (n, m).
#dA is multiplied elementwise with the gradient of ReLU
#cache: dictionary with {"Z", Z}, where Z was the input
#to the activation layer during forward propagation
#Outputs:
#dZ: the derivative of dimension (n,m). It is the elementwise
#product of the derivative of ReLU and dA

    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    Z[Z>0]=1
    Z[Z<=0]=0
    dZ = Z*dA
    return dZ

def linear(Z):
#Computes linear activation of Z
#This function is implemented for completeness
#Inputs:
#Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n'␣dimension
#Outputs:
#A: where A = Linear(Z) is a numpy.ndarray (n, m) representing 'm'␣samples each of 'n' dimension
#cache: a dictionary with {"Z", Z}
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache

def linear_der(dA, cache):
#Computes derivative of linear activation
#This function is implemented for completeness
#Inputs:
#dA: derivative from the subsequent layer of dimension (n, m).
#dA is multiplied elementwise with the gradient of Linear(.)
#cache: dictionary with {"Z", Z}, where Z was the input
#to the activation layer during forward propagation
#Outputs:
#dZ: the derivative of dimension (n,m). It is the elementwise
#product of the derivative of Linear(.) and dA
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
#Computes the softmax activation of the inputs Z
#Estimates the cross entropy loss
#Inputs:
##Z: numpy.ndarray (n, m)
#Y: numpy.ndarray (1, m) of labels
##when y=[] loss is set to []
#Outputs:
#A: numpy.ndarray (n, m) of softmax activations
#cache: a dictionary to store the activations which will be used later␣to estimate derivatives
#loss: cost of prediction

    A = (np.exp(Z-np.max(Z)))/(np.sum(np.exp(Z-np.max(Z)),axis =0,keepdims =␣True))
       cache = {}
       cache["A"] = A
        val1 = 0
        I=0
       val2=0
    if Y!=[]:
        m=Z.shape[1]
        n=Z.shape[0]
        for i in range(1,m+1):
            val1=0
            for k in range(1,n+1):
                if (Y[0][i-1]== k-1):
                    I=1
                else:I=0
                val1 += I * np.log(A[k-1][i-1])
            val2+=val1
        loss =-val2/m
    else:
        loss=[]
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
'''
Computes the derivative of the softmax activation and cross entropy loss
Inputs:
Y: numpy.ndarray (1, m) of labels
cache: a dictionary with cached activations A of size (n,m)
Outputs:
dZ: derivative dL/dZ - a numpy.ndarray of dimensions (n, m)
'''
    A = cache["A"]
    m=Y.shape[1]
    Y_bar = np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[1]):
        dummy = int(Y[0][i])
        Y_bar[dummy][i] = 1
        dZ = (A-Y_bar)/m
    return dZ

def dropout(A, drop_prob, mode='train'):
'''
Using the 'inverted dropout' technique to implement dropout␣regularization.
Inputs:
A: Activation input before dropout is applied - shape is (n,m)
drop_prob: dropout parameter. If drop_prob = 0.3, we drop 30% of␣the neuron activations
mode: Dropout acts differently in training and testing mode. Hence,␣mode is a parameter which
takes in only 2 values, 'train' or 'test'
Outputs:
A: Output of shape (n,m), with some values masked out and other␣values scaled to account for missing values
cache: a tuple which stores the drop_prob, mode and mask for use in␣backward pass.
'''
# When there is no dropout return the same activation
    mask = None
    if drop_prob == 0:
        cache = (drop_prob, mode, mask)
        return A, cache
# The prob_keep is the percentage of activations remaining after dropout
# if drop_out = 0.3, then prob_keep = 0.7, i.e., 70% of the activations␣are retained
    prob_keep = 1-drop_prob
# Note: instead of a binary mask implement a scaled mask, where mask is␣scaled by dividing it
# by the prob_keep for example, if we have input activations of size␣(3,4), then the mask is
# mask = (np.random.rand(3,4)<prob_keep)/prob_keep
# We perform the scaling by prob_keep here so we don't have to do it␣specifically during backpropagation
# We then update A by multiplying it element wise with the mask
    if mode == 'train':
        mask = (np.random.rand(A.shape[0],A.shape[1])< prob_keep)/ prob_keep
        A=A*mask
    elif mode != 'test':
        raise ValueError("Mode value not set correctly, set it to 'train'␣or 'test'")
    cache = (drop_prob, mode, mask)
    return A, cache

def dropout_der(dA_in, cache):
'''
Backward pass for the inverted dropout.
Inputs:
dA_in: derivative from the upper layers of dimension (n,m).
cache: tuple containing (drop_out, mode, mask), where drop_out is␣
,!the probability of drop_out,
if drop_out=0, then the layer does not have any dropout,
mode is either 'train' or 'test' and
mask is a matirx of size (n,m) where 0's indicate masked values
Outputs:
dA_out = derivative of the dropout layer of dimension (n,m)
'''
    dA_out = None
    drop_out, mode, mask = cache
# If there is no dropout return the same derivative from the previous␣layer
    if not drop_out:
        return dA_in
# if mode is 'train' dA_out is dA_in multiplied element wise by mask
# if mode is 'test' dA_out is same as dA_in
    if mode == 'train':
        dA_out = dA_in * mask
    else:
        dA_out = dA_in
    return dA_out

def batchnorm(A, beta, gamma):
'''
Batchnorm normalizes the input A to mean beta and standard deviation gamma
Inputs:
A: Activation input after activation - shape is (n,m), m samples where␣each sample x is (n,1)
beta: mean vector which will be the center of the data after batchnorm␣- shape is (n,1)
gamma: standard deviation vector which will be scale of the data after␣batchnorm - shape (n,1)
Outputs:
Anorm: Normalized version of input A - shape (n,m)
cache: Dictionary of the elements that are necessary for backpropagation
'''
# When there is no batch norm for a layer, the beta and gamma will be empty␣arrays
    if beta.size == 0 or gamma.size == 0:
        cache = {}
        return A, cache
# epsilon value used for scaling during normalization to avoid divide by␣zero.
# don't change this value - the test case will fail if you change this value
    epsilon = 1e-5
    A = A.T
    m,n = A.shape
    mu = (1/m) * np.sum(A,axis = 0)
    xmu = A - mu
    sq = xmu ** 2
    var = (1/m) * np.sum(sq,axis = 0)
    sqrtvar = np.sqrt(var + epsilon)
    ivar = 1./sqrtvar
    xhat = xmu * ivar
    gammax = gamma.T * xhat
    Anorm = gammax + beta.T
#store intermediate
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,epsilon)
    Anorm = Anorm.T
    return Anorm, cache


def batchnorm_der(dA_in, cache):
'''
Derivative of the batchnorm
Inputs:
dA_in: derivative from the upper layers of dimension (n,m).
cache: Dictionary of the elements that are necessary for backpropagation
Outputs:
dA_out: derivative of the batchnorm layer of dimension (n,m)
dbeta: derivative of beta - shape (n,1)
dgamma: derivative of gamma - shape (n,1)
'''
# When the cache is empty, it indicates there was no batchnorm for the layer
    if not cache:
        dbeta = []
        dgamma = []
        return dA_in, dbeta, dgamma

    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    gamma = gamma.T
    dA_in = dA_in.T
    N,D = dA_in.shape
    dbeta = np.sum(dA_in, axis=0,keepdims = True)
    dgammax = dA_in
    dgamma = np.sum(dgammax*xhat, axis=0,keepdims = True)
    dxhat = dgammax * gamma
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. /(sqrtvar**2) * divar
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
    dsq = 1. /N * np.ones((N,D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. /N * np.ones((N,D)) * dmu
    dx = dx1 + dx2
    dA_out = dx.T
    dbeta = dbeta.T
    dgamma = dgamma.T
    return dA_out, dbeta, dgamma

def initialize_network(net_dims, act_list, drop_prob_list):
'''
Initializes the parameters W's and b's of a multi-layer neural network
Adds information about dropout and activations in each layer
Inputs:
net_dims: List containing the dimensions of the network. The values of␣the array represent the number of nodes in
each layer. For Example, if a Neural network contains 784 nodes in the␣input layer, 800 in the first hidden layer,
500 in the secound hidden layer and 10 in the output layer, then␣net_dims = [784,800,500,10].
act_list: list of strings indicating the activation for a layer
drop_prob_list: list of dropout probabilities for each layer
Outputs:
parameters: dictionary of
{"numLayers":..}
activations, {"act1":"..", "act2":"..", ...}
dropouts, {"dropout1": .. , "dropout2": .., ...}
network parameters, {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
The weights are initialized using Kaiming He et al. Initialization
'''
    net_dims_len = len(net_dims)
    parameters = {}
    parameters['numLayers'] = net_dims_len - 1;
    for l in range(net_dims_len-1):
        parameters["act"+str(l+1)] = act_list[l]
        parameters["dropout"+str(l+1)] = drop_prob_list[l]
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1],net_dims[l])*(2./np.sqrt(net_dims[l]))
        parameters["b"+str(l+1)] = np.zeros((net_dims[l+1],1))
    return parameters

def initialize_velocity(parameters, apply_momentum=True):
'''
The function will add Adam momentum parameters, Velocity and␣Gradient-Squares
to the parameters for each of the W's and b's
Inputs:
parameters: dictionary containing,
{"numLayers":..}
activations, {"act1":"..", "act2":"..", ...}
dropouts, {"dropout1": .. , "dropout2": .., ...}
network parameters, {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
Note: It is just one dictionary (parameters) with all these␣key value pairs, not multiple dictionaries
apply_momentum: boolean on whether to apply momentum
Outputs:
parameters: dictionary that has been updated to include velocity and␣Gradient-Squares. It now contains,
{"numLayers":..}
activations, {"act1":"..", "act2":"..", ...}
dropouts, {"dropout1": .. , "dropout2": .., ...}
{"apply_momentum":..}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..],"GdW2":[..],"Gdb2":[..],...}
Note: It is just one dictionary (parameters) with all these␣key value pairs, not multiple dictionaries
'''
    L = parameters['numLayers']
    parameters['apply_momentum'] = apply_momentum
# Initialize Velocity and the Gradient-Squares to zeros the same size as␣the corresponding parameters W's abd b's
    for l in range(L):
        if apply_momentum:
            parameters["VdW" + str(l+1)] = np.zeros_like(parameters["W" +␣str(l+1)])
            parameters["Vdb" + str(l+1)] = np.zeros_like(parameters["b" +␣str(l+1)])
            parameters["GdW" + str(l+1)] = np.zeros_like(parameters["W" +␣str(l+1)])
            parameters["Gdb" + str(l+1)] = np.zeros_like(parameters["b" +␣str(l+1)])
    return parameters

def initialize_bnorm_params(parameters, bnorm_list, apply_momentum):
'''
The function will add batchnorm parameters beta's and gamma's and their␣corresponding
Velocity and Gradient-Squares to the parameters dictionary
Inputs:
parameters: dictionary that contains,
{"numLayers":..}
activations, {"act1":"..", "act2":"..", ...}
dropouts, {"dropout1": .. , "dropout2": .., ...}
{"apply_momentum":..}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..],"GdW2":[..],"Gdb2":[..],...}
Note: It is just one dictionary (parameters) with all these␣key value pairs, not multiple dictionaries
bnorm_list: binary list indicating if batchnorm should be implemented␣for a layer
apply_momentum: boolean on whether to apply momentum
Outputs:
parameters: dictionary that has been updated to include batchnorm␣parameters, beta, gamma
and their corresponding momentum parameters. It now␣contains,
{"numLayers":..}
activations, {"act1":"..", "act2":"..", ...}
dropouts, {"dropout1": .. , "dropout2": .., ...}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..],"GdW2":[..],"Gdb2":[..],...}
{"bnorm_list":..}
batchnorm parameters, {"bnorm_beta1":[..],"bnorm_gamma1":[..],"bnorm_beta2":[..],"bnorm_gamma2":[..],...}
batchnorm velocity parameters, {"Vbnorm_beta1":[..,"Vbnorm_gamma1":[..],"Vbnorm_beta2":[..],"Vbnorm_gamma2":[..],...}
batchnorm Gradient-Square parameters, {"Gbnorm_beta1":[..,"Gbnorm_gamma1":[..],"Gbnorm_beta2":[..],"Gbnorm_gamma2":[..],...}
Note: It is just one dictionary (parameters) with all these␣key value pairs, not multiple dictionaries
'''
    L = parameters['numLayers']
    parameters['bnorm_list'] = bnorm_list
# Initialize batchnorm parameters for the hidden layers only.
# Each hidden layer will have a dictionary of parameters, beta and gamma␣based on the dimensions of the hidden layer.
    for l in range(L):
        if bnorm_list[l]:
            n = parameters["W" + str(l+1)].shape[0]
            parameters['bnorm_beta'+str(l+1)] = np.random.randn(n,1)
            parameters['bnorm_gamma'+str(l+1)] = np.random.randn(n,1)
            if apply_momentum:
                parameters['Vbnorm_beta'+str(l+1)] = np.zeros((n,1))
                parameters['Gbnorm_beta'+str(l+1)] = np.zeros((n,1))
                parameters['Vbnorm_gamma'+str(l+1)] = np.zeros((n,1))
                parameters['Gbnorm_gamma'+str(l+1)] = np.zeros((n,1))
        else:
            parameters['bnorm_beta'+str(l+1)] = np.asarray([])
            parameters['Vbnorm_beta'+str(l+1)] = np.asarray([])
            parameters['Gbnorm_beta'+str(l+1)] = np.asarray([])
            parameters['bnorm_gamma'+str(l+1)] = np.asarray([])
            parameters['Vbnorm_gamma'+str(l+1)] = np.asarray([])
            parameters['Gbnorm_gamma'+str(l+1)] = np.asarray([])
    return parameters

def linear_forward(A_prev, W, b):
'''
Input A_prev propagates through the layer
Z = WA + b is the output of this layer.
Inputs:
A_prev: numpy.ndarray (n,m) the input to the layer
W: numpy.ndarray (n_out, n) the weights of the layer
b: numpy.ndarray (n_out, 1) the bias of the layer
Outputs:
Z: where Z = W.A_prev + b, where Z is the numpy.ndarray (n_out, m)␣
,!dimensions
cache: a dictionary containing the inputs A
'''
    Z = np.dot(W,A_prev) + b
    cache = {}
    cache["A"] = A_prev
    return Z, cache

def layer_forward(A_prev, W, b, activation, drop_prob, bnorm_beta, bnorm_gamma,␣mode):
'''
Input A_prev propagates through the layer followed by activation, batchnorm␣
,!and dropout
Inputs:
A_prev: numpy.ndarray (n,m) the input to the layer
W: numpy.ndarray (n_out, n) the weights of the layer
b: numpy.ndarray (n_out, 1) the bias of the layer
activation: is the string that specifies the activation function
drop_prob: dropout parameter. If drop_prob = 0.3, we drop 30% of the␣
,!neuron activations
bnorm_beta: batchnorm beta
bnorm_gamma: batchnorm gamma
mode: 'train' or 'test' Dropout acts differently in training and␣
,!testing mode. Hence, mode is a parameter which
takes in only 2 values, 'train' or 'test'
Outputs:
A: = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m)␣
,!dimensions
g is the activation function
cache: a dictionary containing the cache from the linear propagation,␣
,!activation, bacthnorm and dropout
to be used for derivative
'''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    A, bnorm_cache = batchnorm(A, bnorm_beta, bnorm_gamma)
    A, drop_cache = dropout(A, drop_prob, mode)
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    cache["bnorm_cache"] = bnorm_cache
    cache["drop_cache"] = drop_cache
    return A, cache

def multi_layer_forward(A0, parameters, mode):
'''
Forward propgation through the layers of the network
Inputs:
A0: numpy.ndarray (n,m) with n features and m samples
parameters: dictionary of network parameters {"W1":[..],"b1":[..],"W2":
,![..],"b2":[..]...}
mode: 'train' or 'test' Dropout acts differently in training and␣
,!testing mode. Hence, mode is a parameter which
takes in only 2 values, 'train' or 'test'
Outputs:
AL: numpy.ndarray (c,m) - outputs of the last fully connected layer␣
,!before softmax
where c is number of categories and m is number of samples
caches: a list of caches from every layer after forward propagation
'''
    L = parameters['numLayers']
    A = A0
    caches = []
    for l in range(L):
        A, cache = layer_forward(A, parameters["W"+str(l+1)],␣parameters["b"+str(l+1)], \
                    parameters["act"+str(l+1)],␣parameters["dropout"+str(l+1)], \
                    parameters['bnorm_beta'+str(l+1)],␣parameters['bnorm_gamma'+str(l+1)], mode)
        caches.append(cache)
    return A, caches

def linear_backward(dZ, cache, W, b):
'''
Backward prpagation through the linear layer
Inputs:
dZ: numpy.ndarray (n,m) derivative dL/dz
cache: a dictionary containing the inputs A, for the linear layer
where Z = WA + b,
Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
W: numpy.ndarray (n,p)
b: numpy.ndarray (n,1)
Outputs:
dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
dW: numpy.ndarray (n,p) the gradient of W
db: numpy.ndarray (n,1) the gradient of b
'''
    A = cache["A"]
    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(dZ,A.T)
    db = np.sum(dZ,axis=1,keepdims=True)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
'''
Backward propagation through the activation and linear layer
Inputs:
dA: numpy.ndarray (n,m) the derivative to the previous layer
cache: dictionary containing the linear_cache and the activation_cache
W: numpy.ndarray (n,p)
b: numpy.ndarray (n,1)
activation: activation of the layer, 'relu' or 'linear'
Outputs:
dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
dW: numpy.ndarray (n,p) the gradient of W
db: numpy.ndarray (n,1) the gradient of b
dbnorm_beta: numpy.ndarray (n,1) derivative of beta for the batchnorm␣
,!layer
dbnorm_gamma: numpy.ndarray (n,1) derivative of gamma for the batchnorm␣
,!layer
'''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    drop_cache = cache["drop_cache"]
    bnorm_cache = cache["bnorm_cache"]
    dA = dropout_der(dA, drop_cache)
    dA, dbnorm_beta, dbnorm_gamma = batchnorm_der(dA, cache["bnorm_cache"])
    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db, dbnorm_beta, dbnorm_gamma

def multi_layer_backward(dAL, caches, parameters):
'''
Back propgation through the layers of the network (except softmax cross␣
,!entropy)
softmax_cross_entropy can be handled separately
Inputs:
dAL: numpy.ndarray (n,m) derivatives from the softmax_cross_entropy␣
,!layer
caches: a dictionary of associated caches of parameters and network␣
,!inputs
parameters: dictionary of network parameters {"W1":[..],"b1":[..],"W2":
,![..],"b2":[..]...}
Outputs:
gradients: dictionary of gradient of network parameters
{"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...\
"dbnorm_beta1":[..],"dbnorm_gamma1":[..],"dbnorm_beta2":[..
,!],"dbnorm_gamma2":[..],...}
'''
    L = len(caches)
    gradients = {}
    dA = dAL
    activation = "linear"   
    for l in reversed(range(L)):
        dA, gradients["dW"+str(l+1)], gradients["db"+str(l+1)], \
        gradients["dbnorm_beta"+str(l+1)], gradients["dbnorm_gamma"+str(l+1)] \
        = layer_backward(dA, caches[l], parameters["W"+str(l+1)],\
        ␣parameters["b"+str(l+1)],parameters["act"+str(l+1)])
    return gradients

def update_parameters_with_momentum_Adam(parameters, gradients, alpha, beta=0.9, beta2=0.99, eps=1e-8):
'''
Updates the network parameters with gradient descent
Inputs:
parameters: dictionary of
network parameters, {"W1":[..],"b1":[..],"W2":[..],"b2":[..
,!],...}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..
,!],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..
,!],"GdW2":[..],"Gdb2":[..],...}
batchnorm parameters, {"bnorm_beta1":[..],"bnorm_gamma1":[..
,!],"bnorm_beta2":[..],"bnorm_gamma2":[..],...}
batchnorm velocity parameters, {"Vbnorm_beta1":[..
,!],"Vbnorm_gamma1":[..],"Vbnorm_beta2":[..],"Vbnorm_gamma2":[..],...}
batchnorm Gradient-Square parameters, {"Gbnorm_beta1":[..
,!],"Gbnorm_gamma1":[..],"Gbnorm_beta2":[..],"Gbnorm_gamma2":[..],...}
and other parameters
::
Note: It is just one dictionary (parameters) with all these␣
,!key value pairs, not multiple dictionaries
gradients: dictionary of gradient of network parameters
{"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
alpha: stepsize for the gradient descent
beta: beta parameter for momentum (same as beta1 in Adam)
beta2: beta2 parameter for Adam
eps: epsilon parameter for Adam
Outputs:
parameters: updated dictionary of
network parameters, {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..
,!],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..
,!],"GdW2":[..],"Gdb2":[..],...}
batchnorm parameters, {"bnorm_beta1":[..],"bnorm_gamma1":[..
,!],"bnorm_beta2":[..],"bnorm_gamma2":[..],...}
batchnorm velocity parameters, {"Vbnorm_beta1":[..
,!],"Vbnorm_gamma1":[..],"Vbnorm_beta2":[..],"Vbnorm_gamma2":[..],...}
batchnorm Gradient-Square parameters, {"Gbnorm_beta1":[..
,!],"Gbnorm_gamma1":[..],"Gbnorm_beta2":[..],"Gbnorm_gamma2":[..],...}
and other parameters
'''
    L = parameters['numLayers']
    apply_momentum = parameters['apply_momentum']
    bnorm_list = parameters['bnorm_list']
    for l in range(L):
        if apply_momentum:
            parameters["VdW" + str(l+1)] = beta*parameters['VdW'+str(l+1)] + (1␣- beta)*gradients["dW"+str(l+1)]
            parameters["Vdb" + str(l+1)] = beta*parameters['Vdb'+str(l+1)] + (1␣- beta)*gradients["db"+str(l+1)]
            parameters["GdW" + str(l+1)] = beta2*parameters['GdW'+str(l+1)] +␣(1 - beta2)*(gradients["dW"+str(l+1)]**2)
            parameters["Gdb" + str(l+1)] = beta2*parameters['Gdb'+str(l+1)] +␣(1 - beta2)*(gradients["db"+str(l+1)]**2)
            parameters["W" + str(l+1)] = parameters['W' + str(l+1)] -␣alpha*parameters['VdW'+str(l+1)]/np.sqrt(parameters['GdW'+str(l+1)] + eps)
            parameters["b" + str(l+1)] = parameters['b' + str(l+1)] -␣alpha*parameters['Vdb'+str(l+1)]/np.sqrt(parameters['Gdb'+str(l+1)] + eps)
        else:
        #When no momentum is required apply regular gradient descent
            parameters["W"+str(l+1)] -= alpha * gradients["dW"+str(l+1)]
            parameters["b"+str(l+1)] -= alpha * gradients["db"+str(l+1)]
        # The Adam momentum for batch norm parameters has been implemented below
        if apply_momentum and bnorm_list[l]:
            parameters['Vbnorm_beta'+str(l+1)] =␣beta*parameters['Vbnorm_beta'+str(l+1)] + \
                (1 -␣beta)*gradients["dbnorm_beta"+str(l+1)]
            parameters['Vbnorm_gamma'+str(l+1)] =␣beta*parameters['Vbnorm_gamma'+str(l+1)] + \
                (1 -␣beta)*gradients["dbnorm_gamma"+str(l+1)]
            parameters['Gbnorm_beta'+str(l+1)] =␣beta2*parameters['Gbnorm_beta'+str(l+1)] + \
                (1 -␣beta2)*(gradients["dbnorm_beta"+str(l+1)]**2)
            parameters['Gbnorm_gamma'+str(l+1)] =␣beta2*parameters['Gbnorm_gamma'+str(l+1)] + \
                (1 -␣beta2)*(gradients["dbnorm_gamma"+str(l+1)]**2)
            parameters['bnorm_beta' + str(l+1)] = parameters['bnorm_beta' +␣str(l+1)] \
                - alpha*parameters['Vbnorm_beta'+str(l+1)]/np.sqrt(parameters['Gbnorm_beta'+str(l+1)] + eps)
            parameters['bnorm_gamma' + str(l+1)] = parameters['bnorm_gamma' +␣str(l+1)] \
                - alpha*parameters['Vbnorm_gamma'+str(l+1)]/np.sqrt(parameters['Gbnorm_gamma'+str(l+1)] + eps)
        elif bnorm_list[l]:
            parameters['bnorm_beta' + str(l+1)] -= alpha *␣gradients["dbnorm_beta"+str(l+1)]
            parameters['bnorm_gamma' + str(l+1)] -= alpha *␣gradients["dbnorm_beta"+str(l+1)]
    return parameters

def multi_layer_network(X, Y, net_dims, act_list, drop_prob_list, bnorm_list,␣num_epochs=3,
batch_size=64, learning_rate=0.2, decay_rate=0.01,␣apply_momentum=True, log=True, log_step=200):
'''
Creates the multilayer network and trains the network
Inputs:
X: numpy.ndarray (n,m) of training data
Y: numpy.ndarray (1,m) of training data labels
net_dims: tuple of layer dimensions
act_list: list of strings indicating the activations for each layer
drop_prob_list: list of dropout probabilities for each layer
bnorm_list: binary list indicating presence or absence of batchnorm for␣
,!each layer
num_epochs: num of epochs to train
batch_size: batch size for training
learning_rate: learning rate for gradient descent
decay_rate: rate of learning rate decay
apply_momentum: boolean whether to apply momentum or not
log: boolean whether to print training progression
log_step: prints training progress every log_step iterations
Outputs:
costs: list of costs (or loss) over training
parameters: dictionary of
network parameters, {"W1":[..],"b1":[..],"W2":[..],"b2":[..
,!],...}
velocity parameters, {"VdW1":[..],"Vdb1":[..],"VdW2":[..
,!],"Vdb2":[..],...}
Gradient-Squares parameters, {"GdW1":[..],"Gdb1":[..
,!],"GdW2":[..],"Gdb2":[..],...}
batchnorm parameters, {"bnorm_beta1":[..],"bnorm_gamma1":[..
,!],"bnorm_beta2":[..],"bnorm_gamma2":[..],...}
batchnorm velocity parameters, {"Vbnorm_beta1":[..
,!],"Vbnorm_gamma1":[..],"Vbnorm_beta2":[..],"Vbnorm_gamma2":[..],...}
batchnorm Gradient-Square parameters, {"Gbnorm_beta1":[..
,!],"Gbnorm_gamma1":[..],"Gbnorm_beta2":[..],"Gbnorm_gamma2":[..],...}
'''
    mode = 'train'
    n, m = X.shape
    parameters = initialize_network(net_dims, act_list, drop_prob_list)
    parameters = initialize_velocity(parameters, apply_momentum)
    parameters = initialize_bnorm_params(parameters, bnorm_list, apply_momentum)
    costs = []
    itr = 0
    for epoch in range(num_epochs):
# estimate stepsize alpha using decay_rate on learning rate using epoch␣number
        alpha = learning_rate*(1/(1+decay_rate*epoch))
        if log:
            print('------- Epoch {} -------'.format(epoch+1))
        for ii in range((m - 1) // batch_size + 1):
            Xb = X[:, ii*batch_size : (ii+1)*batch_size]
            Yb = Y[:, ii*batch_size : (ii+1)*batch_size]
            A0 = Xb
#Forward propogation
            A,caches = multi_layer_forward(A0,parameters,mode)
            AL,softmax_cache,cost=softmax_cross_entropy_loss(A,Yb)
#Backward propogation
            dAL = softmax_cross_entropy_loss_der(Yb,softmax_cache)
            gradients = multi_layer_backward(dAL,caches,parameters)
            parameters =␣update_parameters_with_momentum_Adam(parameters,gradients,alpha)
            if itr % log_step == 0:
                costs.append(cost)
                if log:
                    print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(itr, cost, alpha))
            itr+=1
    return costs, parameters

def classify(X, parameters, mode='test'):
'''
Network prediction for inputs X
Inputs:
X: numpy.ndarray (n,m) with n features and m samples
parameters: dictionary of network parameters
{"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
drop_prob_list: list of dropout probabilities for each layer
mode: 'train' or 'test' Dropout acts differently in training and␣
,!testing mode.
Outputs:
YPred: numpy.ndarray (1,m) of predictions
'''
    A,caches = multi_layer_forward(X,parameters,mode)
    AL,cache,loss = softmax_cross_entropy_loss(A,Y=np.array([]))

    YP=[0]*1
    YP[0]=list(np.argmax(AL,axis =0))
    YPred = np.array(YP)
    return YPred

net_dims = [784, 100, 100, 64, 10] # This network has 4 layers
#784 is for image dimensions
#10 is for number of categories
#100 and 64 are arbitrary
# list of dropout probabilities for each layer
# The length of the list is equal to the number of layers
# Note: Has to be same length as net_dims. 0 indicates no dropout
drop_prob_list = [0, 0, 0, 0]
# binary list indicating if batchnorm should be implemented for a layer
# The length of the list is equal to the number of layers
# 1 indicates bathnorm and 0 indicates no batchnorm

# initialize learning rate, decay_rate and num_iterations
num_epochs = 3
batch_size = 64
learning_rate = 1e-2
decay_rate = 1
apply_momentum = True
np.random.seed(1)
print("Network dimensions are:" + str(net_dims))
print('Dropout= [{}], Batch Size = {}, lr = {}, decay rate = {}'\
.format(drop_prob_list,batch_size,learning_rate,decay_rate))
# getting the subset dataset from MNIST
trX, trY, tsX, tsY = get_mnist()
# We need to reshape the data everytime to match the format (d,m), where d is␣dimensions (784) and m is number of samples
trX = trX.reshape(-1, 28*28).T
trY = trY.reshape(1, -1)
tsX = tsX.reshape(-1, 28*28).T
tsY = tsY.reshape(1, -1)
costs, parameters = multi_layer_network(trX, trY, net_dims, act_list,␣drop_prob_list, bnorm_list, \
num_epochs=num_epochs,␣batch_size=batch_size, learning_rate=learning_rate, \
decay_rate=decay_rate,␣apply_momentum=apply_momentum, log=True)
# compute the accuracy for training set and testing set
train_Pred = classify(trX, parameters)
test_Pred = classify(tsX, parameters)
mask1 = (trY==train_Pred).sum()
trAcc = (mask1/train_Pred.shape[1])*100
mask2 = (tsY==test_Pred).sum()
teAcc = (mask2/test_Pred.shape[1])*100
print("Accuracy for training set is {0:0.3f} %".format(trAcc))
print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
plt.plot(range(len(costs)),costs)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# The following set up gives an accuracy of > 96% for both test and train.
np.random.seed(1)
net_dims = [784, 100, 100, 10]
drop_prob_list = [0, 0, 0]
act_list = ['relu', 'relu', 'linear']
# initialize learning rate, decay_rate and num_iterations
num_epochs = 3
batch_size = 64
learning_rate = 1e-3
decay_rate = 0.1
apply_momentum = True

# getting the subset dataset from MNIST
trX, trY, tsX, tsY = get_mnist()
# We need to reshape the data everytime to match the format (d,m), where d is␣dimensions (784) and m is number of samples
trX = trX.reshape(-1, 28*28).T
trY = trY.reshape(1, -1)
tsX = tsX.reshape(-1, 28*28).T
tsY = tsY.reshape(1, -1)
costs, parameters = multi_layer_network(trX, trY, net_dims, act_list,␣drop_prob_list, bnorm_list, \
num_epochs=num_epochs,
batch_size=batch_size, learning_rate=learning_rate, \
decay_rate=decay_rate,apply_momentum=apply_momentum, log=False)
# compute the accuracy for training set and testing set
train_Pred = classify(trX, parameters)
test_Pred = classify(tsX, parameters)
