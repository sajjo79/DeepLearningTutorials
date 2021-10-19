"""
File edited by: Sajid Iqbal
This file shows the basic working of perceptron.
In addition it uses a training method.
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
    prediction=np.dot(inputs,weights)
    return prediction

def backward_pass(loss,inputs,weights,Y,Y_prime):
    learning_rate=0.01
    de_dw=mse_prime([Y],Y_prime)
    new_weights=weights-(inputs*learning_rate*de_dw)
    return new_weights

def train_perceptron(epochs,X_train,weights,Y_train):
    losses=[]
    for i in range(epochs):
        Y_prime = forward_pass(X_train, weights)
        loss = mse(Y, Y_prime)
        new_weights = backward_pass(loss, X_train, weights, Y_train, Y_prime)
        weights=new_weights
        print('Updated weights and loss  ',weights, loss)
        losses.append(loss)
    plot_loss(losses)

def plot_loss(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()




if __name__=="__main__":
    inputs=np.array([4,5,6])
    weights=np.array([2,1,3])
    Y=50
    train_perceptron(epochs=10,X_train=inputs, weights=weights, Y_train=Y)

