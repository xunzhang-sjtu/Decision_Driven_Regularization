import time
from gurobipy import *
import numpy as np
import pickle
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import RidgeCV

import warnings

class OLS_Related_Estimation:
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

    def OLS_Solver(self,file_path,x_train, z_train):
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
    
    def Lasso_Solver(self,x_train, z_train):
    #     x_train = data[3]
    #     z_train = data[5]
        start = time.time()
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define model
        model = MultiTaskLassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
        # fit model
        warnings.filterwarnings("ignore")
        model.fit(x_train, z_train)
        W_results = model.coef_
        w0_results = model.intercept_
        end = time.time()
        return W_results, w0_results, end-start
    
    def Ridge_Solver(self,x_train, z_train):
    #     x_train = data[3]
    #     z_train = data[5]
        start = time.time()
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define model
        model = RidgeCV(alphas=np.arange(0, 5, 0.1), fit_intercept=True, cv=cv)
        # fit model
        warnings.filterwarnings("ignore")
        model.fit(x_train, z_train)
        W_results = model.coef_
        w0_results = model.intercept_
        end = time.time()
        return W_results, w0_results, end-start