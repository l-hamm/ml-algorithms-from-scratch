import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


# a=np.ones((6,1))
# b=np.zeros((6,1))

# for i in range(np.shape(a)[0]):
#     b[i]=a[i]*i
# #print(a)
# #print(b)
# #print(np.sum(b))

# temp_parameter=4
# theta=np.array([[1,2,3],
#             [4,5,6],
#             [7,8,9],
#             [10,11,12]]
#     )

# X=np.array([[1,2,3],
#             [4,5,6]]
#     )

# #print(np.inner(c,d))
# #print((c * d).sum(axis=1))
# #print(np.einsum('ij,ij->ij', c, d))

# theta_tile=np.tile(theta,(np.shape(X)[0],1))
# print('theta_tile',theta_tile)

# repeat_list=[np.shape(theta)[0]]*np.shape(X)[0]
# print('repeat_list',repeat_list)

# x_repeat=np.repeat(X,repeat_list, axis=0)
# print('x_repeat',x_repeat)

# dot_theta_x=(theta_tile * x_repeat).sum(axis=1)
# print('dot_theta_x',dot_theta_x)

# dot_tau=dot_theta_x/temp_parameter
# print('dot_tau',dot_tau)

# dot_tau_single=np.array(np.hsplit(dot_tau,np.shape(X)[0]))
# print('dot_tau_single',dot_tau_single)

# c=np.max(dot_tau_single,axis=1)
# c=c.reshape(np.shape(c)[0],1)
# print('c',c)

# exp=np.exp(dot_tau_single-c)
# print('exp',exp)

# sum_exp=np.sum(exp,axis=1)
# print('sum_exp',sum_exp)

# trans_exp=np.transpose(exp)
# print('trans_exp',trans_exp)

# h_split_exp=np.array(np.hsplit(trans_exp,np.shape(trans_exp)[1]))
# print(h_split_exp)

# scalar=1/sum_exp
# print('scalar',scalar)

# h=exp*scalar[:, None]
# print('h',h)

#print(np.tile(d,(2,1)))
#print(np.repeat(d, [5, 2], axis=0))

#l=[2]*6
#print(l)

# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# Y = np.arange(0, n)
# theta = np.arange(0, k * d).reshape(k, d)
# lambda_factor=1

# print('X',X)
# print('Y',Y)
# print('theta',theta)

# j=np.arange(0,n)
# print(j)

# condition=np.transpose([np.array(j==Y)])
# print('condition',condition)

# X_filtered=np.multiply(condition,X)
# X_filtered=X_filtered[np.any(X_filtered !=0, axis=1),:]
# print('X_filtered',X_filtered)

# first_term=np.sum(X_filtered)/np.shape(X)[0]
# print('first_term',first_term)

# second_term=lambda_factor/2*np.sum(np.array(theta**2))
# print('second_term',second_term)

# c=first_term+second_term
# print('c',c)

#0:w11,1:w21,2:w12,3:w22,4:w01,5:w02
#w=np.array([0,0,0,0,0,0])
w=np.array([2,2,-2,-2,1,1])
#w=np.array([-2,-2,2,2,1,1])

x1=np.array([-1,-1])
x2=np.array([1,-1])
x3=np.array([-1,1])
x4=np.array([1,1])

x=np.array([x1,x2,x3,x4])

y=np.array([1,-1,-1,1])

f1=np.zeros(x.shape[0])
f2=np.zeros(x.shape[0])
z=np.zeros((4,2))

for i in range(x.shape[0]):
    f1[i]=w[4]+w[0]*x[i,0]+w[1]*x[i,1]
    f2[i]=w[5]+w[2]*x[i,0]+w[3]*x[i,1]
    z[i]=np.array([f1[i],f2[i]])
    
print(z)

#linear activation
linear=5*z-2
print(linear)