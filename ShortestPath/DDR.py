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
        alpha = m.addVars(N,num_nodes,name="alpha")
        expr_obj = 0
        err = []
        for n in range(N):
            cost_true_tem = c_train[n]
            expr_obj = expr_obj + alpha[n,num_nodes-1] - alpha[n,0]
            for ind in range(len(arcs)):
                e = arcs[ind]
                j = e[1]
                i = e[0]
                cost_pred_tem = quicksum([W_ddr[ind,j] * x_train[n,j] for j in range(p)]) + w0_ddr[ind]
                # print("j = ",j,", i = ",i, ", e = ",e)
                m.addConstr(alpha[n,j] - alpha[n,i] >= -mu_fixed*cost_true_tem[ind] - (1-mu_fixed)*cost_pred_tem)
                err.append(cost_true_tem[ind] - cost_pred_tem)
        m.setObjective(quicksum([err[k] * err[k] for k in range(len(err))])/N + lamb*(expr_obj)/N, GRB.MINIMIZE)
        m.optimize()
        W_DDR_rst = m.getAttr('x', W_ddr)
        w0_DDR_rst = m.getAttr('x', w0_ddr)
        W_ddr_val = []
        for i in range(d):
            W_ddr_val.append([W_DDR_rst[(i,j)] for j in range(p)])
        w0_ddr_val = [w0_DDR_rst[i] for i in range(d)]
        # print("w0_DDR_rst = ",w0_ddr_val)
        return w0_ddr_val,W_ddr_val 
    
class run_DDR_Shortest_Path:
    def __init__(self):
        pass
    # from pyepo import EPO
    def obtain_DDR_Cost(self,arcs,w0,W_, grid,dataloader):
        full_shortest_model = My_ShortestPathModel()

        # evaluate
        cost_pred_arr = []
        # load data
        for data in dataloader:
            x, c, w, z = data
            feature = x.numpy()
            # print("Feature Shape = ",np.shape(feature)[0])
            for j in range(np.shape(feature)[0]):
                cost = W_ @ feature[j,:] + w0
                sol_pred = full_shortest_model.solve_Shortest_Path(arcs,cost,grid)
                cost_pred = np.dot(sol_pred, c[j].to("cpu").detach().numpy())
                cost_pred_arr.append(cost_pred)
        # print("Average OLS Cost = ", np.mean(cost_pred_arr))
        return cost_pred_arr
    
    def run(self,DataPath_seed,lamb_arr,mu_arr,arcs,x_train, c_train, grid,loader_test,num_nodes=25):
        ddr_obj = DDR_method()
        # w0_ddr_val,W_ddr_val = ddr_obj.solve_DDR(arcs,lamb,mu,num_nodes,x_train,c_train)
        # cost_DDR_arr = self.obtain_DDR_Cost(arcs,w0_ddr_val,W_ddr_val, grid,loader_test)

        rst_all = {}
        for lamb in lamb_arr:
            # print("======== lambda = ",lamb,"============")
            for mu in mu_arr:
                rst = {}
                # num_nodes = 25
                # w0_ddr_val,W_ddr_val = solve_DDR(lamb,mu,num_nodes,x_train,c_train)
                # cost_DDR = DDR_runner.run(arcs,x_train, c_train, grid,loader_test,lamb,mu,num_nodes)
                w0_ddr_val,W_ddr_val = ddr_obj.solve_DDR(arcs,lamb,mu,num_nodes,x_train,c_train)
                cost_DDR_arr = self.obtain_DDR_Cost(arcs,w0_ddr_val,W_ddr_val, grid,loader_test)
                rst["w0"] = w0_ddr_val
                rst["W"] = W_ddr_val
                rst["cost"] = cost_DDR_arr
                rst_all[lamb,mu] = rst
        with open(DataPath_seed +'rst_DDR.pkl', "wb") as tf:
            pickle.dump(rst_all,tf)

        return rst_all