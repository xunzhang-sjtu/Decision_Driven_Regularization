import time
from gurobipy import *
import numpy as np


class DDR_Method:
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


    def DDR_Solver(self,x_train, z_train, mu, lamb):
    #     x_train = data[3]
        # z_train_min = np.minimum(z_train,thres) 
        N,p = x_train.shape
        N,d = z_train.shape

        # DDR
        m = Model("ddr")
        #m.setParam("DualReductions",0)
        m.setParam('OutputFlag', 0)

        W_ind = tuplelist( [(i,j) for i in range(d) for j in range(p)] )
        w0_ind = tuplelist( [i for i in range(d)] )
        t_ind = tuplelist( [n for n in range(N)] )
        
        W_ddr = m.addVars(W_ind, lb=-GRB.INFINITY )
        w0_ddr = m.addVars(w0_ind, lb=-GRB.INFINITY )
        t_ddr = m.addVars( t_ind, lb=-GRB.INFINITY )

        m.setObjective( self.Loss(x_train, z_train, W_ddr, w0_ddr) + lamb*(quicksum([t_ddr[n]  for n in range(N)])/ N) , GRB.MINIMIZE)

        for n in range(N):
            for i in range(d):
                m.addConstr( (- mu*z_train[n,i] - (1- mu)*(quicksum([x_train[n][j]*W_ddr[i,j] for j in range(p)]) + w0_ddr[i])-\
                        t_ddr[n] <= 0))
        m.optimize()
        W = m.getAttr('x', W_ddr)
        w0 = m.getAttr('x', w0_ddr)
        t = m.getAttr('x', t_ddr)
        W_results = []
        for i in range(d):
            W_results.append([W[(i,j)] for j in range(p)])
        w0_results = [w0[i] for i in range(d)]
    #     t_results = [t[n] for n in range(N)]
        return W_results, w0_results