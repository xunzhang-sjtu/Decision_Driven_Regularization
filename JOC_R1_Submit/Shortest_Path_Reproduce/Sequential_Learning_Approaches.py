import random
import numpy as np
import torch
import os
import pathlib
import pickle
import pandas as pd
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from Performance import performance_evaluation
perfs = performance_evaluation()

from Performance import H2h_Regret_Evaluation
h2h_regret_eva = H2h_Regret_Evaluation()

from Data import Data_Simulator
DS_Obj = Data_Simulator()

from Oracle import Oracle_Processing
Oracle_Proc = Oracle_Processing()

from OLS import OLS_Processing
OLS_Proc = OLS_Processing()

from DDR import DDR_Processing
DDR_Proc = DDR_Processing()

from PYEPO import EPO_Processing
PYEPO_Proc = EPO_Processing()

from Random_Forest import RF_Processing
RF_Proc = RF_Processing()

from XGBoost_SP import XG_Processing
XG_Proc = XG_Processing()

from Data_Load_Store import Load_Store_Methods
Data_LSM = Load_Store_Methods()


from sklearn.preprocessing import PolynomialFeatures

def Run_Oracle(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)

        cost_Oracle_Post_all,cost_Oracle_Ante_all = Oracle_Proc.Implement_Oracle(arcs, grid,mis,bump,\
                                                                    W_star_all,x_test_all,c_test_all,\
                                                                    iteration_all,num_feat,data_generation_process)
        with open(DataPath+'cost_Oracle_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_Oracle_Ante_all,tf)

def Run_OLS(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)

        cost_OLS_Post_all,cost_OLS_Ante_all,RMSE_in_all,RMSE_out_all = OLS_Proc.Implement_OLS(arcs, grid,mis,bump,\
                                                                                                W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                                                                iteration_all,num_feat,data_generation_process)
        with open(DataPath+'cost_OLS_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_OLS_Ante_all,tf)
        with open(DataPath+'RMSE_in_OLS_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_OLS_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)

def Run_Random_Forest(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)

        cost_RF_Post_all,cost_RF_Ante_all,RMSE_in_all,RMSE_out_all = RF_Proc.Implement_RF(arcs, grid,mis,bump,\
                                                            W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                            iteration_all,num_feat,data_generation_process)
        with open(DataPath+'cost_RF_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_RF_Ante_all,tf)
        with open(DataPath+'RMSE_in_RF_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_RF_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)

def Run_XGBoost(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)

        cost_XG_Post_all,cost_XG_Ante_all,RMSE_in_all,RMSE_out_all = XG_Proc.Implement_XG(arcs, grid,mis,bump,\
                                                            W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                            iteration_all,num_feat,data_generation_process)
        with open(DataPath+'cost_XG_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_XG_Ante_all,tf)
        with open(DataPath+'RMSE_in_XG_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_XG_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)


def transform_quadratic_data(x_input_all,iteration_all):
    x_input_quad_all = {}
    for iter in iteration_all:
        x_input = x_input_all[iter]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(x_input)
        x_input_quad_all[iter] = X_poly
    return x_input_quad_all

def Run_OLS_Quadratic(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)

        x_train_quad_all = transform_quadratic_data(x_train_all,iteration_all)
        x_test_quad_all = transform_quadratic_data(x_test_all,iteration_all)

        cost_OLS_Post_all,cost_OLS_Ante_all,RMSE_in_all,RMSE_out_all = OLS_Proc.Implement_OLS_quad(arcs, grid,mis,bump,\
                                                                                                W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                                                                iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all)
        with open(DataPath+'cost_OLS_Ante_quad_all.pkl', "wb") as tf:
            pickle.dump(cost_OLS_Ante_all,tf)
        with open(DataPath+'RMSE_in_OLS_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_OLS_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)


def Run_Random_Forest_Quadratic(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
        x_train_quad_all = transform_quadratic_data(x_train_all,iteration_all)
        x_test_quad_all = transform_quadratic_data(x_test_all,iteration_all)

        cost_RF_Post_all,cost_RF_Ante_all,RMSE_in_all,RMSE_out_all = RF_Proc.Implement_RF_quad(arcs, grid,mis,bump,\
                                                            W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                            iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all)
        with open(DataPath+'cost_RF_Ante_quad_all.pkl', "wb") as tf:
            pickle.dump(cost_RF_Ante_all,tf)
        with open(DataPath+'RMSE_in_RF_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_RF_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)

def Run_XGBoost_Quadratic(DataPath,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
        x_train_quad_all = transform_quadratic_data(x_train_all,iteration_all)
        x_test_quad_all = transform_quadratic_data(x_test_all,iteration_all)

        cost_XG_Post_all,cost_XG_Ante_all,RMSE_in_all,RMSE_out_all = XG_Proc.Implement_XG_quad(arcs, grid,mis,bump,\
                                                            W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                            iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all)
        with open(DataPath+'cost_XG_Ante_quad_all.pkl', "wb") as tf:
            pickle.dump(cost_XG_Ante_all,tf)
        with open(DataPath+'RMSE_in_XG_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_in_all,tf)
        with open(DataPath+'RMSE_out_XG_quad_all.pkl', "wb") as tf:
            pickle.dump(RMSE_out_all,tf)