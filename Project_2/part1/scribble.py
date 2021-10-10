import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


a=np.ones((6,1))
b=np.zeros((6,1))

for i in range(np.shape(a)[0]):
    b[i]=a[i]*i
#print(a)
#print(b)
#print(np.sum(b))

temp_parameter=4
theta=np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]]
    )

X=np.array([[1,2,3],
            [4,5,6]]
    )

#print(np.inner(c,d))
#print((c * d).sum(axis=1))
#print(np.einsum('ij,ij->ij', c, d))

theta_tile=np.tile(theta,(np.shape(X)[0],1))
print('theta_tile',theta_tile)

repeat_list=[np.shape(theta)[0]]*np.shape(X)[0]
print('repeat_list',repeat_list)

x_repeat=np.repeat(X,repeat_list, axis=0)
print('x_repeat',x_repeat)

dot_theta_x=(theta_tile * x_repeat).sum(axis=1)
print('dot_theta_x',dot_theta_x)

dot_tau=dot_theta_x/temp_parameter
print('dot_tau',dot_tau)

dot_tau_single=np.array(np.hsplit(dot_tau,np.shape(X)[0]))
print('dot_tau_single',dot_tau_single)

c=np.max(dot_tau_single,axis=1)
c=c.reshape(np.shape(c)[0],1)
print('c',c)

exp=np.exp(dot_tau_single-c)
print('exp',exp)

sum_exp=np.sum(exp,axis=1)
print('sum_exp',sum_exp)

trans_exp=np.transpose(exp)
print('trans_exp',trans_exp)

h_split_exp=np.array(np.hsplit(trans_exp,np.shape(trans_exp)[1]))
print(h_split_exp)

scalar=1/sum_exp
print('scalar',scalar)

h=exp*scalar[:, None]
print('h',h)

#print(np.tile(d,(2,1)))
#print(np.repeat(d, [5, 2], axis=0))

#l=[2]*6
#print(l)