import numpy as np

## Some common function below
def Loss(x_train, z_train, W, w0):
    #W and w0 can be a tuplelist or an array
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    a = []
    for n in range(N):
        for i in range(d):
            temp = []
            for j in range(p):
                temp.append(x_train[n][j]*W[i,j])
            a.append((z_train[n][i] - sum(temp) - w0[i])*(z_train[n][i] - sum(temp) - w0[i]))      
    return np.sum(a)/N

def Loss1(x_train, z_train, W, w0,w0_ols):
    #W and w0 can only be an array
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    W = np.array(W)
    w0 = np.array(w0)
    return np.sum(np.square(z_train - x_train @ W.T - np.tile(w0_ols, (N,1))))/N

def L1_Loss1(x_train, z_train, W, w0,w0_ols):
    #W and w0 can only be an array
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    W = np.array(W)
    w0 = np.array(w0)
    return np.sum(np.absolute(z_train - x_train @ W.T - np.tile(w0_ols, (N,1))))/N