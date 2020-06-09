import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle

np.random.seed(1)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- channels - in this case 1
    n_y -- scalar, number of classes
        s
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
    return X, Y

def initialize_parameters():
    """
    Returns:
    parameters -- dictionary containing W1, W2
    """
        
    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def forward_propagation(X, parameters, net={}):
    """
    shallow ccnn

    conv,relu,mp,conv,relu,mp,fc
    
    Arguments:
    X -- (height, width)
    parameters -- dict containing weights

    Returns:
    Z3
    """

    W1 = parameters['W1']
    W2 = parameters['W2']
    
    net['Z1'] = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    net['A1'] = tf.nn.relu(net['Z1'])
    net['P1'] = tf.nn.max_pool(net['A1'], ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
    net['Z2'] = tf.nn.conv2d(net['P1'], W2, strides=[1, 1, 1, 1], padding='SAME')
    net['A2'] = tf.nn.relu(net['Z2'])
    net['P2'] = tf.nn.max_pool(net['A2'], ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
    net['P'] = tf.contrib.layers.flatten(net['P2'])
    net['Z3'] = tf.contrib.layers.fully_connected(net['P'], 2, activation_fn=None)

    return net['Z3']

def compute_cost(Z3, Y):
    """
    Arguments:
    Z3 -- output of forward prop
    Y -- "true" labels
    
    Returns:
    cost
    """

    cost = tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost

def main():
    tf.get_logger().setLevel(3)
    dims = 64
    normalized = True

    classes = ["normal","sars-cov-2"]
    x_train, y_train, x_test, y_test = load_dataset(dims)

    _, _, parameters = model(x_train, y_train, x_test, y_test)

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=1, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3                                         
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Init params
    parameters = initialize_parameters()
    
    net ={}

    # Forward propagation
    Z3 = forward_propagation(X, parameters, net)
    
    # sigmoid cross entropy
    cost = compute_cost(Z3, Y)
    
    # backprop adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
     
    with tf.Session() as sess:
    
        sess.run(init)
        
        for epoch in range(num_epochs):

            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train})

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, temp_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(temp_cost)
            # minibatch_cost = 0.
            # num_minibatches = int(m / minibatch_size)
            # seed = seed + 1
            # minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # for minibatch in minibatches:

            #     (minibatch_X, minibatch_Y) = minibatch

            #     _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
            #     minibatch_cost += temp_cost / num_minibatches
                
            # # print(Z3.eval({X: X_train, Y: Y_train}))
            # print(cost.eval({X: X_train, Y: Y_train}))

            # # Print the cost every epoch
            # if print_cost == True and epoch % 5 == 0:
            #     print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            # if print_cost == True and epoch % 1 == 0:
            #     costs.append(minibatch_cost)
        
        # for val in net.values():
        #     print(val.shape)

        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
        
    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # partition
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def load_dataset(dims):

    # add d to use distorted aug dataset
    # add b to use resampled aug dataset
    x_train = pickle.load(open("./covid/" + str(dims) + "/x_train_d.p","rb"))
    y_train = pickle.load(open("./covid/" + str(dims) + "/y_train_d.p","rb"))
    x_test = pickle.load(open("./covid/" + str(dims) + "/x_test.p","rb"))
    y_test = pickle.load(open("./covid/" + str(dims) + "/y_test.p","rb"))

    x_train = np.asarray(x_train) / 255
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test) / 255
    y_test = np.asarray(y_test)

    y_train = one_hot(y_train,2)
    y_test = one_hot(y_test,2)
    # one hot encoding

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train, x_test, y_test

def one_hot(arr,nc):
    res = np.zeros((arr.shape[0],nc))
    # assume nc is correctly provided

    for i,val in enumerate(arr):
        res[i][val] = 1

    return res

if __name__ == '__main__':
    main()

