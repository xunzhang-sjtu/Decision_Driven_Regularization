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
from xgboost import XGBRegressor

class XG_Processing:
    def __init__(self):
        pass
    def Implement_XG(self,arcs, grid,mis,bump,W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process):
        
        cost_XG_Post = {}; cost_XG_Ante = {}; RMSE_in_all = {}; RMSE_out_all = {}
        for iter in iteration_all:
            xg_model = XGBRegressor(n_estimators=2, random_state=42)
            xg_model.fit(x_train_all[iter], c_train_all[iter])
            cost_dem = xg_model.predict(x_test_all[iter])
            cost_in = xg_model.predict(x_train_all[iter])
            RMSE_in_all[iter] = np.sqrt(np.sum((cost_in - c_train_all[iter])**2)/len(cost_in[:,0]))
            RMSE_out_all[iter] = np.sqrt(np.sum((cost_dem - c_test_all[iter])**2)/len(cost_dem[:,0]))



            if data_generation_process == "SPO_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                # cost_OLS_Post[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Post(arcs, grid,cost_dem,cost_oracle_pred,noise_test_all[iter])
                cost_XG_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_dem,cost_oracle_pred)

            if data_generation_process == "DDR_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                cost_oracle_pred = (cost_oracle_ori ** mis).T
                cost_XG_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_dem,cost_oracle_pred)

            if iter % 20 == 0 and iter > 0:
                # print("OLS: iter=",iter,",cost_OLS_Post =",np.nanmean(cost_OLS_Post[iter]),",cost_OLS_Ante=",np.nanmean(cost_OLS_Ante[iter]))
                print("XGboost: iter=",iter,",cost_XG_Ante=",np.nanmean(cost_XG_Ante[iter]))
        return cost_XG_Post,cost_XG_Ante,RMSE_in_all,RMSE_out_all