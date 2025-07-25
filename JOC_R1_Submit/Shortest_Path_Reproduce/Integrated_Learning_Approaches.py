import random
import numpy as np
import torch
import os
import pathlib
import pickle
import pandas as pd
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import Figure_H2H_Regret

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

from Data_Load_Store import Load_Store_Methods
Data_LSM = Load_Store_Methods()

from sklearn.preprocessing import PolynomialFeatures


def Run_DDR(DataPath,mu_all,lamb_all,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)


    cost_DDR_Post_all,cost_DDR_Ante_all,RMSE_in_all,RMSE_out_all = DDR_Proc.Implement_DDR(mu_all,lamb_all,arcs, grid,mis,bump,\
                                                                W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                                    iteration_all,num_feat,data_generation_process)

    with open(DataPath+'cost_DDR_Ante_all.pkl', "wb") as tf:
        pickle.dump(cost_DDR_Ante_all,tf)
    with open(DataPath+'RMSE_in_DDR_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_DDR_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)

def run_EPO_approaches(DataPath,method_names,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
    batch_size = 20
    num_epochs = 1000
    cost_EPO_Post_all,cost_EPO_Ante_all,RMSE_in_all,RMSE_out_all = PYEPO_Proc.Implement_EPO(DataPath,iteration_all,batch_size,num_epochs,method_names,\
                                                W_star_all,bump,x_train_all,c_train_all,x_test_all,c_test_all,\
                                                arcs,grid,perfs,num_feat,mis,data_generation_process)

    with open(DataPath+'cost_'+method_names[0]+'_Ante_all.pkl', "wb") as tf:
        pickle.dump(cost_EPO_Ante_all,tf)
    with open(DataPath+'RMSE_in_'+method_names[0]+'_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_'+method_names[0]+'_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)


def transform_quadratic_data(x_input_all,iteration_all):
    x_input_quad_all = {}
    for iter in iteration_all:
        x_input = x_input_all[iter]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(x_input)
        x_input_quad_all[iter] = X_poly
    return x_input_quad_all


def Run_DDR_Quadratic(DataPath,mu_all,lamb_all,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
    x_train_quad_all = transform_quadratic_data(x_train_all,iteration_all)
    x_test_quad_all = transform_quadratic_data(x_test_all,iteration_all)

    cost_DDR_Post_all,cost_DDR_Ante_all,RMSE_in_all,RMSE_out_all = DDR_Proc.Implement_DDR_quad(mu_all,lamb_all,arcs, grid,mis,bump,\
                                                                W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                                    iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all)

    with open(DataPath+'cost_DDR_Ante_quad_all.pkl', "wb") as tf:
        pickle.dump(cost_DDR_Ante_all,tf)
    with open(DataPath+'RMSE_in_DDR_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_DDR_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)


def run_EPO_approaches_Quadratic(DataPath,method_names,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
    x_train_quad_all = transform_quadratic_data(x_train_all,iteration_all)
    x_test_quad_all = transform_quadratic_data(x_test_all,iteration_all)

    batch_size = 20
    num_epochs = 1000
    cost_EPO_Post_all,cost_EPO_Ante_all,RMSE_in_all,RMSE_out_all = PYEPO_Proc.Implement_EPO_quad(DataPath,iteration_all,batch_size,num_epochs,method_names,\
                                                W_star_all,bump,x_train_all,c_train_all,x_test_all,c_test_all,\
                                                arcs,grid,perfs,num_feat,mis,data_generation_process,x_train_quad_all,x_test_quad_all)

    with open(DataPath+'cost_'+method_names[0]+'_Ante_quad_all.pkl', "wb") as tf:
        pickle.dump(cost_EPO_Ante_all,tf)
    with open(DataPath+'RMSE_in_'+method_names[0]+'_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_'+method_names[0]+'_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)


def add_self_quadratic_data(x_input_all,iteration_all):
    x_input_quad_all = {}
    for iter in iteration_all:
        x_input = x_input_all[iter]
        squared_data = x_input ** 2
        extended_data = np.hstack((x_input, squared_data))
        x_input_quad_all[iter] = extended_data
    return x_input_quad_all


def Run_DDR_self_Quadratic(DataPath,mu_all,lamb_all,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
    x_train_quad_all = add_self_quadratic_data(x_train_all,iteration_all)
    x_test_quad_all = add_self_quadratic_data(x_test_all,iteration_all)

    cost_DDR_Post_all,cost_DDR_Ante_all,RMSE_in_all,RMSE_out_all = DDR_Proc.Implement_DDR_quad(mu_all,lamb_all,arcs, grid,mis,bump,\
                                                                W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,\
                                                                    iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all)

    with open(DataPath+'cost_DDR_Ante_self_quad_all.pkl', "wb") as tf:
        pickle.dump(cost_DDR_Ante_all,tf)
    with open(DataPath+'RMSE_in_DDR_self_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_DDR_self_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)

def run_EPO_approaches_self_Quadratic(DataPath,method_names,arcs, grid,mis,bump,iteration_all,num_feat,data_generation_process):
        
    x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all = Data_LSM.load_input_data(DataPath)
    x_train_quad_all = add_self_quadratic_data(x_train_all,iteration_all)
    x_test_quad_all = add_self_quadratic_data(x_test_all,iteration_all)

    batch_size = 20
    num_epochs = 1000
    cost_EPO_Post_all,cost_EPO_Ante_all,RMSE_in_all,RMSE_out_all = PYEPO_Proc.Implement_EPO_quad(DataPath,iteration_all,batch_size,num_epochs,method_names,\
                                                W_star_all,bump,x_train_all,c_train_all,x_test_all,c_test_all,\
                                                arcs,grid,perfs,num_feat,mis,data_generation_process,x_train_quad_all,x_test_quad_all)

    with open(DataPath+'cost_'+method_names[0]+'_Ante_self_quad_all.pkl', "wb") as tf:
        pickle.dump(cost_EPO_Ante_all,tf)
    with open(DataPath+'RMSE_in_'+method_names[0]+'_self_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_in_all,tf)
    with open(DataPath+'RMSE_out_'+method_names[0]+'_self_quad_all.pkl', "wb") as tf:
        pickle.dump(RMSE_out_all,tf)