import time
from gurobipy import *
import numpy as np
import pickle
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import RidgeCV
from Shortest_Path_Model import My_ShortestPathModel

import warnings

class DDR_method:
    def __init__(self):
        pass
    def solve_DDR(self,arcs,lamb,mu_fixed,num_nodes,x_train,c_train):
        N,p = x_train.shape
        N,d = c_train.shape
        # print("Num of data = ",N, ",num of feature = ", p, ", num of acrs = ", d)
        
        # DDR
        m = Model("ddr")
        #m.setParam("DualReductions",0)
        m.setParam('OutputFlag', 0)

        W_ind = tuplelist( [(i,j) for i in range(d) for j in range(p)] )
        w0_ind = tuplelist( [i for i in range(d)])

        W_ddr = m.addVars( W_ind, lb=-GRB.INFINITY,name = "W" )
        w0_ddr = m.addVars( w0_ind, lb=-GRB.INFINITY,name = "W0" )
        alpha = m.addVars(N,num_nodes,lb=-GRB.INFINITY,name="alpha")
        expr_obj = 0
        err = []
        for n in range(N):
            cost_true_tem = c_train[n]
            expr_obj = expr_obj + alpha[n,num_nodes-1] - alpha[n,0]
            for ind in range(len(arcs)):
                cost_pred_tem = quicksum([W_ddr[ind,j] * x_train[n,j] for j in range(p)]) + w0_ddr[ind]
                err.append(cost_true_tem[ind] - cost_pred_tem)

                e = arcs[ind]
                j = e[1]
                i = e[0]
                # print("j = ",j,", i = ",i, ", e = ",e)
                m.addConstr(alpha[n,j] - alpha[n,i] >= -mu_fixed*cost_true_tem[ind] - (1-mu_fixed)*cost_pred_tem)
                
        m.setObjective(quicksum([err[k] * err[k] for k in range(len(err))])/N + lamb*(expr_obj)/N, GRB.MINIMIZE)
        m.optimize()
        W_DDR_rst = m.getAttr('x', W_ddr)
        w0_DDR_rst = m.getAttr('x', w0_ddr)
        W_ddr_val = []
        for i in range(d):
            W_ddr_val.append([W_DDR_rst[(i,j)] for j in range(p)])
        w0_ddr_val = [w0_DDR_rst[i] for i in range(d)]
        # print("w0_DDR = ",np.round(w0_ddr_val,4))
        # print("DDR Obj val = ", m.objVal)

        return w0_ddr_val,W_ddr_val 
    
class run_DDR_Shortest_Path:
    def __init__(self):
        pass

    def run(self,DataPath_seed,lamb_arr,mu_arr,arcs,grid,num_nodes=25):
        with open(DataPath_seed+'Data.pkl', "rb") as tf:
            Data = pickle.load(tf)
        x_test = Data["x_test"]
        c_test = Data["c_test"]
        x_train = Data["x_train"]
        c_train = Data["c_train"]

        ddr_obj = DDR_method()
        from Peformance import performance_evaluation
        perfs = performance_evaluation()

        rst_all = {}
        for lamb in lamb_arr:
            # print("======== lambda = ",lamb,"============")
            for mu in mu_arr:
                rst = {}
                w0_ddr_val,W_ddr_val = ddr_obj.solve_DDR(arcs,lamb,mu,num_nodes,x_train,c_train)
                cost_DDR_arr = perfs.compute_Cost_with_Prediction(arcs,w0_ddr_val,W_ddr_val, grid,c_test,x_test)

                rst["w0"] = w0_ddr_val
                rst["W"] = W_ddr_val
                rst["cost"] = cost_DDR_arr
                rst_all[lamb,mu] = rst
                print("lambda = ",lamb,", mu = ",mu, ", Average DDR cost = ",np.nanmean(cost_DDR_arr))
        with open(DataPath_seed +'rst_DDR.pkl', "wb") as tf:
            pickle.dump(rst_all,tf)

        return rst_all