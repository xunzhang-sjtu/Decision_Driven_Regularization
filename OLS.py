import time
from gurobipy import *
import numpy as np
import pickle

class ols_method:
    def __init__(self):
        pass

    def Loss(self,x_train, z_train, W, w0):
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




    def ols_solver(self,file_path,x_train, z_train):
    #     x_train = data[3]
    #     z_train = data[5]
        N,p = x_train.shape
        N,d = z_train.shape
        
        start = time.time()
        m = Model("OLS")
        #m.setParam("DualReductions",0)
        m.setParam('OutputFlag', 0)

        W_ind = tuplelist([(i,j) for i in range(d) for j in range(p)])
        w0_ind = tuplelist([i for i in range(d)])
        
        W_ddr = m.addVars(W_ind, lb=-GRB.INFINITY)
        
        w0_ddr = m.addVars(w0_ind, lb=-GRB.INFINITY)

        m.setObjective(self.Loss(x_train, z_train, W_ddr, w0_ddr), GRB.MINIMIZE)

        m.optimize()
        end = time.time()

        W = m.getAttr('x', W_ddr)
        w0 = m.getAttr('x', w0_ddr)
        W_results = []
        for i in range(d):
            W_results.append([W[(i,j)] for j in range(p)])
        w0_results = [w0[j] for j in range(d)]
        end = time.time()

        return W_results, w0_results, end-start, m.ObjVal