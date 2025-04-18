import random
import numpy as np
import torch
import os
import pathlib
import pickle
from gurobipy import *
from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
from rsome import cpt_solver as cpt
import pandas as pd
torch.manual_seed(42)
torch.cuda.manual_seed(42)

from Performance import performance_evaluation
perfs = performance_evaluation()

def Prepare_Data(DataPath,lower, upper, p, d, coef_seed,iteration_all,num_test, num_train, alpha,mis,data_generation_process,x_dist, e_dist, x_low, x_up, x_mean, x_var, bump):
# #  ****** Coef generation *********
    from Data import data_generation
    data_gen = data_generation()
    # W_star = data_gen.generate_truth(DataPath,lower, upper, p, d, coef_seed,data_generation_process) 
    # print("W_star = ",W_star[0,:])
    np.random.seed(coef_seed)
    x_test_all = {}; c_test_all = {}; x_train_all = {}; c_train_all = {}; W_star_all = {}; noise_train_all = {}; noise_test_all = {}
    for iter in iteration_all:
        DataPath_iter = DataPath +"iter="+str(iter)+"/"
        pathlib.Path(DataPath_iter).mkdir(parents=True, exist_ok=True)
        W_star = data_gen.generate_truth(DataPath_iter,lower, upper, p, d, iter,data_generation_process) 
        # #  ****** Data generation *********
        x_test_all[iter], c_test_all[iter], x_train_all[iter], c_train_all[iter], noise_train_all[iter],noise_test_all[iter],W_star_all[iter] = data_gen.generate_samples(iter,DataPath_iter,p, d, num_test, num_train, alpha, W_star, mis, num_test, 
                                data_generation_process, x_dist, e_dist, x_low, x_up, x_mean, x_var, bump) 
        # print()
    return x_test_all, c_test_all, x_train_all, c_train_all, noise_train_all,noise_test_all,W_star_all

def Implement_Oracle(arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,iteration_all,num_feat,data_generation_process):
    cost_Oracle_with_noise_all = {}; cost_Oracle_wo_noise_all = {}
    for iter in iteration_all:
        if data_generation_process == "SPO_Data_Generation":
            cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
            cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
            cost_Oracle_with_noise_all[iter] = perfs.compute_SPO_out_of_sample_Cost(arcs, grid,cost_oracle_pred,cost_oracle_pred,noise_test_all[iter])
            # print("Oracle: iter=",iter,",cost_Oracle_with_noise_all=",np.nanmean(cost_Oracle_with_noise_all[iter]))

        if data_generation_process == "DDR_Data_Generation":
            cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
            cost_oracle_pred = (cost_oracle_ori ** mis).T
            cost_Oracle_with_noise_all[iter] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_oracle_pred,cost_oracle_pred,noise_test_all[iter],True)
            cost_Oracle_wo_noise_all[iter] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_oracle_pred,cost_oracle_pred,noise_test_all[iter],False)

            # print("Oracle: iter=",iter,",cost_Oracle_with_noise_all=",np.nanmean(cost_Oracle_with_noise_all[iter]),",cost_Oracle_wo_noise_all=",np.nanmean(cost_Oracle_wo_noise_all[iter]))
    return cost_Oracle_with_noise_all,cost_Oracle_wo_noise_all

def Implement_OLS(arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process):
    from OLS import ols_method
    ols_method_obj = ols_method()
    W_ols_all = {}; w0_ols_all = {}; t_ols_all = {}; obj_ols_all = {}
    cost_OLS_with_noise_all = {}; cost_OLS_wo_noise_all = {}
    for iter in iteration_all:
        # compute OLS performance
        W_ols_all[iter], w0_ols_all[iter], t_ols_all[iter], obj_ols_all[iter] = ols_method_obj.ols_solver("",x_train_all[iter], c_train_all[iter])
        cost_dem = (W_ols_all[iter] @ x_test_all[iter].T).T + w0_ols_all[iter]

        if data_generation_process == "SPO_Data_Generation":
            cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
            cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
            cost_OLS_with_noise_all[iter] = perfs.compute_SPO_out_of_sample_Cost(arcs, grid,cost_dem,cost_oracle_pred,noise_test_all[iter])

        if data_generation_process == "DDR_Data_Generation":
            cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
            cost_oracle_pred = (cost_oracle_ori ** mis).T
            cost_OLS_with_noise_all[iter] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_dem,cost_oracle_pred,noise_test_all[iter],True)
            cost_OLS_wo_noise_all[iter] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_dem,cost_oracle_pred,noise_test_all[iter],False)
        if iter % 20 == 0 and iter > 0:
            print("OLS: iter=",iter,",cost_OLS_with_noise_all =",np.nanmean(cost_OLS_with_noise_all[iter]))
    return cost_OLS_with_noise_all,cost_OLS_wo_noise_all

def Implement_DDR(mu_all,lamb_all,arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process):
    from DDR import DDR_method
    ddr_object = DDR_method()
    num_nodes = grid[0] * grid[0]

    w0_ddr_dict = {}; W_ddr_dict = {}
    cost_DDR_with_noise_all = {}; cost_DDR_wo_noise_all = {}
    for iter in iteration_all:
        for mu in mu_all:
            for lamb in lamb_all:
                w0_ddr_dict[iter,mu,lamb],W_ddr_dict[iter,mu,lamb],alpha_rst,obj_ddr = ddr_object.solve_DDR(arcs,lamb,mu,num_nodes,x_train_all[iter],c_train_all[iter])
                cost_pred = (W_ddr_dict[iter,mu,lamb] @ x_test_all[iter].T).T + w0_ddr_dict[iter,mu,lamb]

                if data_generation_process == "SPO_Data_Generation":
                    cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                    cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                    cost_DDR_with_noise_all[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter])

                if data_generation_process == "DDR_Data_Generation":
                    cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                    cost_oracle_pred = (cost_oracle_ori ** mis).T
                    cost_DDR_with_noise_all[iter,mu,lamb] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter],True)
                    cost_DDR_wo_noise_all[iter,mu,lamb] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter],False)
        if iter % 1 == 0 and iter > 0:
            print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_with_noise_all =",np.nanmean(cost_DDR_with_noise_all[iter,mu,lamb]))

    return cost_DDR_with_noise_all,cost_DDR_wo_noise_all



def cross_compare2plus(c_item, c_base, c_oracle):
    N = len(c_item)
    c_diff = c_base - c_item
    lbel = np.zeros((N,1))
    
    equals = np.sum(c_diff == 0)
    wins = np.sum(c_diff > 0) # indicate num of c_item is lower than c_base
    lose = np.sum(c_diff < 0)
    
    lbel[c_diff < 0] = 1
    lbel[c_diff > 0] = -1
    
#     print(N, equals, wins, lose)
    if N == equals:
        win_ratio = 0.5
    else:
        win_ratio = wins/(N - equals)
    cost_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base))
    regret_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base) - np.mean(c_oracle))
    return win_ratio, cost_reduction, regret_reduction

grid = (2,2) # grid size
from Network import network_design
Network = network_design()
arcs,arc_index_mapping = Network._getArcs(grid)

num_train = 100 # number of training data
num_feat = 5 # size of feature
num_test = 1000
deg = 1.0 # polynomial degree
e = 0.5 # scale of normal std or the range of uniform. For the error term

lower = 0 # coef lower bound
upper = 1 # coef upper bound
p = num_feat # num of features
d = (grid[0] - 1) * (grid[0] - 1) * 2 + 2 * (grid[0] - 1) # num of arcs
num_nodes = grid[0]*grid[0]
alpha = e # scale of normal std or the range of uniform. For the error term
mis = deg # model misspecification
# coef_seed = 2

x_dist = 'uniform'
e_dist = 'normal'
x_low = -2
x_up = 2
x_mean = 2
x_var = 2
bump = 100
iteration_all = np.arange(0,100)
mu_all = np.round(np.arange(0.1,1.0,0.1),4)
lamb_all = np.round(np.arange(0.0,1.0,0.1),4)

data_generation_process = "SPO_Data_Generation"
# data_generation_process = "DDR_Data_Generation"

coef_seed_all = np.arange(3,100)
for coef_seed in coef_seed_all:
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    grandparent_directory = os.path.dirname(parent_directory)
    DataPath = os.path.dirname(grandparent_directory) + '/Data/Shortest_Path_Reproduce/'+str(grid[0])+'by'+str(grid[1])+'_grid_' + data_generation_process + "/"
    pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)
    print("grandparent_directory:", grandparent_directory)
    
    DataPath = DataPath + "data_size="+str(num_train)+"_deg="+str(deg)+"_e="+str(e)+"_d="+str(d)+"_x_dist="+x_dist+"_num_test="+str(num_test)+"_coef_seed="+str(coef_seed)+"_diff_W/"
    pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)
    
    print("DataPath:", DataPath)
    
    x_test_all, c_test_all, x_train_all, c_train_all,noise_train_all,noise_test_all,W_star_all = Prepare_Data(DataPath,lower, upper, p, d, coef_seed,iteration_all,num_test, num_train, alpha,mis,data_generation_process,x_dist, e_dist, x_low, x_up, x_mean, x_var, bump)
    
    cost_Oracle_with_noise_all,cost_Oracle_wo_noise_all = Implement_Oracle(arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,iteration_all,num_feat,data_generation_process)

    cost_OLS_with_noise_all,cost_OLS_wo_noise_all = Implement_OLS(arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process)

    cost_DDR_with_noise_all,cost_DDR_wo_noise_all = Implement_DDR(mu_all,lamb_all,arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process)

    with open(DataPath+'cost_OLS_with_noise_all.pkl', "wb") as tf:
        pickle.dump(cost_OLS_with_noise_all,tf)
    with open(DataPath+'cost_Oracle_with_noise_all.pkl', "wb") as tf:
        pickle.dump(cost_Oracle_with_noise_all,tf)
    with open(DataPath+'cost_DDR_with_noise_all.pkl', "wb") as tf:
        pickle.dump(cost_DDR_with_noise_all,tf)
    # with open(DataPath+'cost_SPO_all.pkl', "wb") as tf:
#     pickle.dump(cost_SPO_all,tf)

    h2h_ddr_vs_ols_all = {}; cost_reduction_ddr_vs_ols_all = {}; regret_reduction_ddr_vs_ols_all = {}
    h2h_ddr_vs_ols_avg = np.zeros((len(mu_all),len(lamb_all))); regret_ddr_vs_ols_avg = np.zeros((len(mu_all),len(lamb_all)))

    mu_index = 0
    for mu in mu_all:
        lamb_index = 0
        for lamb in lamb_all:

            h2h_ddr_ols = np.zeros(len(iteration_all)); cost_reduction_ddr_vs_ols = np.zeros(len(iteration_all)); regret_reduction_ddr_vs_ols = np.zeros(len(iteration_all))
            for iter_index in range(len(iteration_all)):
                iter = iteration_all[iter_index]
                h2h_ddr_ols[iter_index],cost_reduction_ddr_vs_ols[iter_index],regret_reduction_ddr_vs_ols[iter_index] = cross_compare2plus(cost_DDR_with_noise_all[iter,mu,lamb], cost_OLS_with_noise_all[iter], cost_Oracle_with_noise_all[iter])
            h2h_ddr_vs_ols_avg[mu_index,lamb_index] = np.nanmean(h2h_ddr_ols)
            regret_ddr_vs_ols_avg[mu_index,lamb_index] = np.nanmean(regret_reduction_ddr_vs_ols)
            print("mu=",mu,",lamb=",lamb,"h2h_ddr_ols=",np.nanmean(h2h_ddr_ols),"regret=",np.nanmean(regret_reduction_ddr_vs_ols))
            lamb_index = lamb_index + 1
        
            h2h_ddr_vs_ols_all[mu,lamb] = h2h_ddr_ols
            cost_reduction_ddr_vs_ols_all[mu,lamb] = cost_reduction_ddr_vs_ols
            regret_reduction_ddr_vs_ols_all[mu,lamb] = regret_reduction_ddr_vs_ols
        mu_index = mu_index + 1

    regret_DDR_vs_OLS_para_avg_df = pd.DataFrame(regret_ddr_vs_ols_avg)
    regret_DDR_vs_OLS_para_avg_df.index = ["$\mu="+str(mu)+"$" for mu in mu_all]
    regret_DDR_vs_OLS_para_avg_df.columns = ["$\lambda="+str(lamb)+"$" for lamb in lamb_all]
    regret_DDR_vs_OLS_para_avg_df.to_csv(DataPath+"regret_ddr_vs_ols_avg.csv")

    h2h_DDR_vs_OLS_para_avg_df = pd.DataFrame(h2h_ddr_vs_ols_avg)
    h2h_DDR_vs_OLS_para_avg_df.index = ["$\mu="+str(mu)+"$" for mu in mu_all]
    h2h_DDR_vs_OLS_para_avg_df.columns = ["$\lambda="+str(lamb)+"$" for lamb in lamb_all]
    h2h_DDR_vs_OLS_para_avg_df.to_csv(DataPath+"h2h_ddr_vs_ols_avg.csv")

    print()