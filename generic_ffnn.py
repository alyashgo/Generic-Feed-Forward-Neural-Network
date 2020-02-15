# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:02:48 2020

@author: Yashank Singh
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv("C:\\Users\\Yashank Singh\\Desktop\\assn1_ELL409\\mnist_train.csv", header=None)
train_set = train_data.values

test_data = pd.read_csv("C:\\Users\\Yashank Singh\\Desktop\\assn1_ELL409\\mnist_test.csv", header=None)
test_set = test_data.values
X_test = test_set.T
X_test = X_test

m=train_set.shape[0]
n=train_set.shape[1]-1
X=train_set[:,1:].reshape(m,n)
Y=np.zeros((m,10))
indices = train_set[:,0]
Y[np.arange(Y.shape[0]),indices]=1 
X_train = X[0:6000].T
Y_train = Y[0:6000].T
X_cross_valid = X[6000:].T
Y_cross_valid = Y[6000:].T
X= X.T
Y=Y.T
#converting one training example into a column
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_grad(x):
	return np.multiply(sigmoid(x),1-sigmoid(x))
	
def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x), axis=0)
    return s

def relu(x):
    r = np.maximum(0,x)
    return r

def relu_grad(x):
    grad = np.zeros(x.shape)
    grad[x>0]=1     #index by comparing values >0, and put 1 there 
    return grad
    
def initialize_params(layer_dims):
    
    params = {}
    for i in range(1,len(layer_dims)):
        n_prev = layer_dims[i-1]
        n_curr = layer_dims[i]
        W = np.random.randn(n_curr, n_prev) * 0.02
        b = np.zeros((n_curr, 1))
        params['W'+str(i)]= W
        params['b'+str(i)]= b
    
    return params

def compute_cost(A,Y, params, Lambda, layer_dims):
    """
    Computes the cross-entropy cost 
    Arguments:
    A -- The softmax output of the final layer, of shape (10, number of examples)
    Y -- "true" labels vector of shape (10, number of examples)
    """
    m = Y.shape[1]
    reg_cost = 0.0
    for i in range(1, len(layer_dims)):
    	w = params['W'+str(i)]
    	reg_cost+= np.sum(np.square(w))	
    # MSE LOSS cost = (1/m)*np.sum(np.square(A-Y))+ (Lambda/(2*m))*reg_cost
    cost = (-1/m)*(np.sum(np.sum(np.multiply(Y,np.log(A)),axis=0,keepdims=True),axis=1,keepdims=True)) + (Lambda/(2*m))*reg_cost 
    cost = np.squeeze(cost)  #makes cost into a float 
    return cost
    
def forward_prop(X, params, activation_func, layer_dims):
    """
    Argument:
    X- input data of size (nl-1*m) nl=input dim, m = training examples
    W- weight matrix of size nl*nl-1
    b- bias is of size nl*1 
    activation function to be used 
    
    Returns:
    cache -- a dictionary containing "Z", "A"
    """
    func_list={"sigmoid" : sigmoid, "relu" : relu}
    cache ={}
    cache['A0']=X
    for i in range(1,len(layer_dims)-1):                        
        W= params['W'+str(i)]                                 #retrieve Weights connecting layers 
        b = params['b'+str(i)]                                #retrieve bias      
        z = np.dot(W,cache['A'+str(i-1)]) +b;                 #z is of size nl*m
        a = func_list[activation_func](z)
        cache['Z'+str(i)]=z
        cache['A'+str(i)]=a
    
    #for the softmax layer 
    W = params['W'+str(len(layer_dims)-1)]
    b = params['b'+str(len(layer_dims)-1)]
    z = np.dot(W,cache['A'+str(len(layer_dims)-2)]) +b;
    a = softmax(z)
    cache['Z'+str(len(layer_dims)-1)] = z
    cache['A'+str(len(layer_dims)-1)] = a
    
    return cache  

def update_params(params, num_layer, learning_rate, grads):
    params['W'+str(num_layer)] = params['W'+str(num_layer)] - learning_rate*grads['dW'+str(num_layer)]  
    params['b'+str(num_layer)] = params['b'+str(num_layer)] - learning_rate*grads['db'+str(num_layer)]
    
    return params
    
     
def back_prop(Y, cache, params, learning_rate, Lambda, activation_func, layer_dims):
    """
    Arguments:
    cache -- contains all Ai's and Zi's, where A0==X
    Returns:
    grads -- dZ_l, dW_l, db_l, dA_l-1, 
    Note : A_0 = X
    """
    func_list_grad = {"sigmoid" : sigmoid_grad, "relu" : relu_grad}
    grads = {}
    L = len(layer_dims)
    m = Y.shape[1]
    #for the softmax layer-------------------------------------------------------------------------------
    grads['dZ'+str(L-1)] = cache['A'+str(L-1)] - Y          #dZfinal  = Afinal - Ytrue //for softmax
    grads['dW'+str(L-1)] = (1/m)*np.dot(grads['dZ'+str(L-1)], cache['A'+str(L-2)].T) + (Lambda/m)*params['W'+str(L-1)]
    grads['db'+str(L-1)] = (1/m)*np.sum(grads['dZ'+str(L-1)],axis=1,keepdims =True)
    grads['dA'+str(L-2)] = np.dot(params['W'+str(L-1)].T,grads['dZ'+str(L-1)])
    params = update_params(params, L-1, learning_rate, grads)
    #for the hidden layers--------------------------------------------------------------------------------   
    for i in range(L-2,0,-1):
        grads['dZ'+str(i)] = np.multiply(grads['dA'+str(i)],func_list_grad[activation_func](cache['Z'+str(i)]))
        grads['dW'+str(i)] = (1/m)*np.dot(grads['dZ'+str(i)], cache['A'+str(i-1)].T) + (Lambda/m)*params['W'+str(i)]
        grads['db'+str(i)] = (1/m)*np.sum(grads['dZ'+str(i)],axis=1,keepdims =True)
        grads['dA'+str(i-1)] = np.dot(params['W'+str(i)].T,grads['dZ'+str(i)])
        params = update_params(params, i, learning_rate, grads)           #updating parameters
    
    return params

def accuracy(A, Y):
    pred = np.argmax(A, axis=0)
    Y_act = np.argmax(Y, axis=0)
    return np.sum(np.asarray(pred==Y_act, np.int))/len(pred)

def model(X_train,Y_train,learning_rate, activation_func, layer_dims, num_iter, Lambda, batch_size):
    params= initialize_params(layer_dims)
    COST=[]
    J=[]
    plt.figure()
    for j in range(num_iter):
        for i in range(int(X_train.shape[1]/batch_size)):
            X = X_train[:,i*batch_size:(i+1)*batch_size]
            Y = Y_train[:,i*batch_size:(i+1)*batch_size]
            cache = forward_prop(X, params, activation_func, layer_dims)
            cost = compute_cost(cache['A'+str(len(layer_dims)-1)], Y, params, Lambda, layer_dims)
            params = back_prop(Y, cache, params, learning_rate, Lambda, activation_func, layer_dims) 
        COST.append(cost)
        J.append(j)
        print(cost)
        if(j==num_iter-1):
            print("acc on training set= "+str(accuracy(cache['A'+str(len(layer_dims)-1)], Y)))
    plt.plot(J,COST, label = str(learning_rate)+", "+str(Lambda)+", "+str(activation_func)+", "+str(layer_dims)+", "+str(num_iter))
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.legend()
    plt.show()
    return params          

def predict(X_test, X_train, Y_train, X_cross_valid, Y_cross_valid, learning_rate, activation_func, layer_dims, num_iter, Lambda, batch_size):
    params =model(X_train, Y_train, learning_rate, activation_func, layer_dims, num_iter, Lambda, batch_size) 
    cache = forward_prop(X_cross_valid, params, activation_func, layer_dims)
    print("acc on cross validation set= "+str(accuracy(cache['A'+str(len(layer_dims)-1)], Y_cross_valid)))
    cache = forward_prop(X_test, params, activation_func, layer_dims) 
    A = cache['A'+str(len(layer_dims)-1)]          
    Y_pred = np.argmax(A, axis=0)
    out = pd.DataFrame(Y_pred)
    out.index+=1
    out.to_csv("C:\\Users\\Yashank Singh\\Desktop\\assn1_ELL409\\predict.csv")
    
    return Y_pred

predict(X_test, X_train, Y_train, X_cross_valid, Y_cross_valid, 0.003, "relu", [784,256,256,256,10],10, 0.8,50)        
#0.02, "relu", [784,100,100,100,10],300, 0.07, 7000 BEST ENTRY   95.3%    
#predict(X_test, X_train, Y_train, X_cross_valid, Y_cross_valid, 0.01, "relu", [784,150,100,100,10],10, 0.2,60)  95.2%
#0.002, "relu", [784,256,256,10],15, 0.2,50 95.4%