import time
from gurobipy import *
import numpy as np
import pickle
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import RidgeCV
# from Shortest_Path_Model import My_ShortestPathModel

import warnings

class ols_method:
    def __init__(self):
        pass

    def Loss(self,x_train, z_train, W, w0):
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
    
    def lasso_solver(self,x_train, z_train):
        start = time.time()
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        model = MultiTaskLassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
        warnings.filterwarnings("ignore")
        model.fit(x_train, z_train)
        W_results = model.coef_
        w0_results = model.intercept_
        end = time.time()
        return W_results, w0_results, end-start
    
    def ridge_solver(self,x_train, z_train):
        start = time.time()
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        model = RidgeCV(alphas=np.arange(0, 5, 0.1), fit_intercept=True, cv=cv)
        warnings.filterwarnings("ignore")
        model.fit(x_train, z_train)
        W_results = model.coef_
        w0_results = model.intercept_
        end = time.time()
        return W_results, w0_results, end-start
    
# class run_OLS_Shortest_Path:
#     def __init__(self):
#         pass
    
#     def run(self,DataPath_seed,arcs,grid):
#         with open(DataPath_seed+'Data.pkl', "rb") as tf:
#             Data = pickle.load(tf)
#         x_test = Data["x_test"]
#         c_test = Data["c_test"]
#         x_train = Data["x_train"]
#         c_train = Data["c_train"]
    
#         ols_method_obj = ols_method()
#         W_ols, w0_ols, t_ols, obj_ols = ols_method_obj.ols_solver("",x_train, c_train)
#         # print("w0_ols = ",np.round(w0_ols,4))
#         # print("OLS objective value = ",obj_ols)


#         from Peformance import performance_evaluation
#         perfs = performance_evaluation()
#         cost_OLS = perfs.compute_Cost_with_Prediction(arcs,w0_ols,W_ols, grid,c_test,x_test)

#         rst = {"w0":w0_ols,"W":W_ols,"cost":cost_OLS}
#         with open(DataPath_seed +'rst_OLS.pkl', "wb") as tf:
#             pickle.dump(rst,tf)
#         return cost_OLS