import numpy as np

Wfh, Wfx, bf, Wch=0,0,-100,-100
Wih, Wix, bi, Wcx=0,100,100,50
Woh, Wox, bo, bc=0,100,0,0
x=np.array((1,0,1))

f=np.zeros(x.shape[0])
i=np.zeros(x.shape[0])
o=np.zeros(x.shape[0])
c=np.zeros(x.shape[0])
h=np.zeros(x.shape[0])

for t in range(x.shape[0]):
    if t==0:
        h_prev=0
        c_prev=0
    else:
        h_prev=h[t-1]
        c_prev=c[t-1]

    f[t]=round(1/(1 + np.exp(-(Wfh*h_prev+Wfx*x[t]+bf))))
    i[t]=round(1/(1 + np.exp(-(Wih*h_prev+Wix*x[t]+bi))))
    o[t]=round(1/(1 + np.exp(-(Woh*h_prev+Wox*x[t]+bo))))
    c[t]=f[t]*c_prev+i[t]*round(np.tanh(Wch*h_prev+Wcx*x[t]+bc))
    h[t]=o[t]*round(np.tanh(c[t]))

print(h)



t=1
x=3
w1=0.01
w2=-5
b=-1

z1=w1*x
a1=z1 * (z1 > 0)
z2=w2*a1+b
y=1/(1+np.exp(-z2))
C=1/2*(y-t)**2

print('z1',z1)
print('a1',a1)
print('z2',z2)
print('y',y)
print('C',C)

dz1_dw1=x
da1_dz1=1
dz2_da1=w2
dy_dz2=np.exp(-z2)/(1+np.exp(-z2))**2
dC_dy=-(y-t)

dC_dw1=x*1*w2*(np.exp(-z2)/(1+np.exp(-z2))**2)*(-(y-t))
dC_dw2=a1*(np.exp(-z2)/(1+np.exp(-z2))**2)*(-(y-t))
dC_db=1*(np.exp(-z2)/(1+np.exp(-z2))**2)*(-(y-t))

print('dC_dw1',dC_dw1)
print('dC_dw2',dC_dw2)
print('dC_db',dC_db)

print('dC_dw1',dz1_dw1*da1_dz1*dz2_da1*dy_dz2*dC_dy)