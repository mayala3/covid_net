import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import pickle
from sklearn.metrics import balanced_accuracy_score

def main():
    dims = 64
    normalized = True

    classes = ["normal","covid-19"]
    x_train, y_train, x_test, y_test = load_dataset(dims)

    if normalized:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    # plt.imshow(x_train[:,0].reshape((dims,dims)))
    # plt.title(classes[int(y_train[0][0])])
    # plt.savefig("./covid/plots/" + str(dims) + "_normal_sample")
    # plt.clf()

    # plt.imshow(x_train[:,5253].reshape((dims,dims)))
    # plt.title(classes[int(y_train[0][5253])])
    # plt.savefig("./covid/plots/" + str(dims) + "_covid_sample")
    # plt.clf()

    d = model(x_train, y_train, x_test, y_test, dims, classes, num_iterations = 1000, learning_rate = 0.005, print_cost = True)
    
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.savefig("./covid/plots/" + str(dims) + "_learn")

def load_dataset(dims):
    x_train = pickle.load(open("./covid/" + str(dims) + "/x_train_d.p","rb"))
    y_train = pickle.load(open("./covid/" + str(dims) + "/y_train_d.p","rb"))
    x_test = pickle.load(open("./covid/" + str(dims) + "/x_test.p","rb"))
    y_test = pickle.load(open("./covid/" + str(dims) + "/y_test.p","rb"))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    y_train = y_train.reshape(y_train.shape[0],1).T
    y_test = y_test.reshape(y_test.shape[0],1).T

    # flatten x
    x_train = x_train.reshape(x_train.shape[0],-1).T
    x_test = x_test.reshape(x_test.shape[0],-1).T

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train, x_test, y_test

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1.0 / (np.exp(-z) + 1)
    ### END CODE HERE ###
    
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0.0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    epsilon = 0.00001
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    Z = np.dot(w.T,X) + b
    A = sigmoid(Z)                                 # compute activation
    cost = - np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon)) / m                         # compute cost
    ### END CODE HERE ###
    
#     print(A.shape)
#     print(Y.shape)
#     print(cost.shape)
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = np.dot(X,(A - Y).T) / m
    db = np.sum(A - Y) / m
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T,X) + b)
    ### END CODE HERE ###
    
#     for i in range(A.shape[1]):
        
#         # Convert probabilities A[0,i] to actual predictions p[0,i]
#         ### START CODE HERE ### (≈ 4 lines of code)
        
#         ### END CODE HERE ###
    Y_prediction = np.round(A, decimals = 0)
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, dims, classes, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    print(np.sum(Y_train))

    print(np.sum(Y_test))

    # Y_testz = zip(Y_prediction_test,Y_test)

    # for (a,b) in Y_testz:
    #     print(b)

    # Y_testz_cov = [(a,b) for (a,b) in Y_testz if b == 1]

    # Y_test_cov = np.asarray([b for (a,b) in Y_testz_cov])
    # Y_pred_cov = np.asarray([a for (a,b) in Y_testz_cov])

    # print(len(Y_pred_cov))

    # Y_testz_noncov = [(a,b) for (a,b) in Y_testz if b == 0]

    # Y_test_noncov = np.asarray([b for (a,b) in Y_testz_noncov])
    # Y_pred_noncov = np.asarray([a for (a,b) in Y_testz_noncov])

    tp = 0
    tn = 0
    fn = 0
    fp = 0
    
    for i, pred in enumerate(Y_prediction_test[0]):
        truth = Y_test[0][i]

        if truth == 0:
            if pred == 0:
                tn += 1
            else:
                fp += 1
        else:
            if pred == 1:
                tp += 1
            else:
                fn += 1
    
    print("cov test accuracy: {} %".format((tp/(tp+fn)) * 100))
    print("noncov test accuracy: {} %".format((tn/(tn+fp)) * 100))

    print("balanced test accuracy: {} %".format(0.5 * ((tn/(tn+fp)) + (tp/(tp+fn))) * 100))

    # for i, pred in enumerate(Y_prediction_test[0]):
    #     if Y_test[0][i] != pred:
    #         plt.imshow(X_test[:,i].reshape((dims,dims)))
    #         plt.title("y = " + classes[int(Y_test[0][i])] + ", you predicted that it is a \"" + classes[int(pred)] +  "\" picture.")
    #         plt.savefig("./plots/" + str(dims) + "_misclassified")
    #         plt.clf()
    #         break
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

if __name__ == "__main__":
    main()