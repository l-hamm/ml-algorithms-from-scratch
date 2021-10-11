import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """

    # #Repeat theta n-times
    # theta_tile=np.tile(theta,(np.shape(X)[0],1))

    # #Define numbers of repitions of X
    # repeat_list=[np.shape(theta)[0]]*np.shape(X)[0]

    # #Repeat X k-times
    # x_repeat=np.repeat(X,repeat_list, axis=0)

    # #Compute pair-wise dot product of theta_tile and X_repeat and divide by tau
    # dot_theta_x_tau=((theta_tile * x_repeat).sum(axis=1))/temp_parameter

    # #Split array into n sub-arrays (one array for each X)
    # dot_tau_single=np.array(np.hsplit(dot_theta_x_tau,np.shape(X)[0]))

    # #Compute c
    # c=np.max(dot_tau_single,axis=1)
    # c=c.reshape(np.shape(c)[0],1)

    # #Compute exponent
    # exp=np.exp(dot_tau_single-c)

    # #Compute sum of exponents
    # sum_exp=np.sum(exp,axis=1)

    # #Compute scalar part of h
    # scalar=1/sum_exp

    # #Compute h
    # h=np.transpose(exp*scalar[:, None])

    # Compute the matrix of theta*X' (each row is a category, column an example)
    R = (theta.dot(X.T))/temp_parameter
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    
    # Compute H matrix
    h = np.exp(R - c)
    
    # Divide H by the normalizing term
    h = h/np.sum(h, axis = 0)


    
    return h



def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """

    n = np.shape(X)[0]
    
    #Create condition matrix
    condition=np.identity(n) 

    # Compute probabilities
    probabilities=compute_probabilities(X, theta, temp_parameter)

    #Clip probabilities of 0 and 1
    probabilities_clipped = np.clip(probabilities, 0.0000000000001, 0.9999999999999)
    
    #Log of probabilities
    log_probabilities_clipped = np.log(probabilities_clipped)
    
    #Only keep first n rows in log
    #first_log_probabilities=log_probabilities_clipped[:n,:]

    condition = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(np.shape(theta)[0],n)).toarray()

    # Result
    first_term = (-1/n)*np.sum(log_probabilities_clipped[condition == 1])     
    second_term = (lambda_factor/2)*np.linalg.norm(theta)**2

    c=first_term+second_term
    
    return c




def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n=np.shape(X)[0]
    k=np.shape(theta)[0]
    
    #Compute probabilities
    probabilities=compute_probabilities(X,theta,temp_parameter)

    #Condition matrix
    condition=sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()

    #Substract probabilities from condition matrix
    diff=condition-probabilities
    
    #Result
    theta_new=(-1/(temp_parameter*n))*np.matmul(diff,X)+lambda_factor*theta
    theta_new=theta-alpha*theta_new
    
    return theta_new


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    
    return (np.mod(train_y, 3), np.mod(test_y, 3))

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    # classified_labels=get_classification(X, theta, temp_parameter)
    
    # classified_labels_mod3=np.mod(classified_labels,3)

    # result=np.array(classified_labels_mod3==Y)

    # error_abs=np.size(result)-np.sum(result)

    # error_percent=error_abs/np.size(result)
    

    # return error_percent
    y_pred=get_classification(X,theta,temp_parameter)
    return 1-(np.mod(y_pred,3)==Y).mean()


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
