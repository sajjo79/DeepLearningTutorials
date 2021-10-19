"""
File edited by: Sajid Iqbal
This file shows the basic working of perceptron.
the script is easy for deep learning beginners
comedxd@gmail.com
"""
import numpy as np
#====================================================
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    n=len(y_true)
    err=-2 * (y_true-y_pred) / n
    return err

def forward_pass(inputs,weights):
    #print(weights)
    prediction=np.dot(inputs,weights)
    print('prediction',prediction)
    return prediction

def backward_pass(inputs,weights,Y,Y_prime):
    learning_rate=0.01
    de_dw=mse_prime([Y],Y_prime)
    new_weights=weights-(inputs*learning_rate*de_dw)
    #print(de_dw)
    #print(new_weights)
    return new_weights



if __name__=="__main__":
    inputs=np.array([4,5,6])
    weights=np.array([2,1,3])
    Y=50
    # iteration 1
    Y_prime=forward_pass(inputs,weights)
    loss=mse(Y,Y_prime)
    new_weights=backward_pass(inputs,weights,Y,Y_prime)
    weights=new_weights
    print('loss and updated weights',loss, weights)
    # iteration 2
    Y_prime = forward_pass(inputs, weights)
    loss = mse(Y, Y_prime)
    new_weights = backward_pass(inputs, weights, Y, Y_prime)
    weights = new_weights
    print('loss and updated weights',loss, weights)

    #iteration 3
    Y_prime = forward_pass(inputs, weights)
    loss = mse(Y, Y_prime)
    new_weights = backward_pass(inputs, weights, Y, Y_prime)
    weights = new_weights
    print('loss and updated weights',loss, weights)

    # iteration 4
    Y_prime = forward_pass(inputs, weights)
    loss = mse(Y, Y_prime)
    new_weights = backward_pass(inputs, weights, Y, Y_prime)
    weights = new_weights
    print('loss and updated weights', loss, weights)

    # iteration 5
    Y_prime = forward_pass(inputs, weights)
    loss = mse(Y, Y_prime)
    new_weights = backward_pass(inputs, weights, Y, Y_prime)
    weights = new_weights
    print('loss and updated weights', loss, weights)
