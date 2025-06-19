import time
from gurobipy import *
import numpy as np
import pickle
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import RidgeCV
# from Shortest_Path_Model import My_ShortestPathModel
from Performance import performance_evaluation
perfs = performance_evaluation()
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



class OLS_Processing:
    def __init__(self):
        self.ols_method_obj = ols_method()

    def Implement_OLS(self,arcs, grid,mis,bump,W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process):
        ols_method_obj = self.ols_method_obj
        
        W_ols_all = {}; w0_ols_all = {}; t_ols_all = {}; obj_ols_all = {}
        cost_OLS_Post = {}; cost_OLS_Ante = {}; RMSE_in_all = {}; RMSE_out_all = {}
        for iter in iteration_all:
            # compute OLS performance
            W_ols_all[iter], w0_ols_all[iter], t_ols_all[iter], obj_ols_all[iter] = ols_method_obj.ols_solver("",x_train_all[iter], c_train_all[iter])
            cost_dem = (W_ols_all[iter] @ x_test_all[iter].T).T + w0_ols_all[iter]

            cost_in = (W_ols_all[iter] @ x_train_all[iter].T).T + w0_ols_all[iter]
            RMSE_in_all[iter] = np.sqrt(np.sum((cost_in - c_train_all[iter])**2)/len(cost_in[:,0]))
            RMSE_out_all[iter] = np.sqrt(np.sum((cost_dem - c_test_all[iter])**2)/len(cost_dem[:,0]))

            if data_generation_process == "SPO_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                # cost_OLS_Post[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Post(arcs, grid,cost_dem,cost_oracle_pred,noise_test_all[iter])
                cost_OLS_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_dem,cost_oracle_pred)

            if data_generation_process == "DDR_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                cost_oracle_pred = (cost_oracle_ori ** mis).T
                cost_OLS_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_dem,cost_oracle_pred)

            if iter % 20 == 0 and iter > 0:
                # print("OLS: iter=",iter,",cost_OLS_Post =",np.nanmean(cost_OLS_Post[iter]),",cost_OLS_Ante=",np.nanmean(cost_OLS_Ante[iter]))
                print("OLS: iter=",iter,",cost_OLS_Ante=",np.nanmean(cost_OLS_Ante[iter]))
        return cost_OLS_Post,cost_OLS_Ante,RMSE_in_all,RMSE_out_all