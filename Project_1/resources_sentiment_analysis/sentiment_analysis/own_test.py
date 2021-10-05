import project1 as p1
import utils
import numpy as np

T=1
L=0.2

train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
val_feature_matrix = np.array([[1, 1], [2, -1]])
train_labels = np.array([1, -1, 1])
val_labels = np.array([-1, 1])


classifier_accuracy(p1.average_perceptron,train_feature_matrix,val_feature_matrix,train_labels,val_labels,T=2)