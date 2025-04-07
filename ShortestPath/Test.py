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

data_generation_process = "SPO_Data_Generation"
# data_generation_process = "DDR_Generation"


# import pyepo
# generate data
grid = (3,3) # grid size
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
coef_seed = 1

x_dist = 'uniform'
e_dist = 'normal'
x_low = -2
x_up = 2
x_mean = 2
x_var = 2
bump = 100

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
DataPath = os.path.dirname(grandparent_directory) + '/Data/Shortest_Path_'+str(grid[0])+'by'+str(grid[1])+'_grid_0406_' + data_generation_process + "/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)
print("grandparent_directory:", grandparent_directory)
print("DataPath:", DataPath)
DataPath = DataPath + "data_size="+str(num_train)+"_deg="+str(deg)+"_e="+str(e)+"_d="+str(d)+"_x_dist="+x_dist+"_coef_seed="+str(coef_seed)+"_diff_W_exp=1/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)

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


def _getArcs(grid):
    arcs = []
    for i in range(grid[0]):
        # edges on rows
        for j in range(grid[1] - 1):
            v = i * grid[1] + j
            arcs.append((v, v + 1))
        # edges in columns
        if i == grid[0] - 1:
            continue
        for j in range(grid[1]):
            v = i * grid[1] + j
            arcs.append((v, v + grid[1]))

    arc_index_mapping = {}
    for i in range(len(arcs)):
        arc = arcs[i]
        arc_index_mapping[arc] = i

    return arcs,arc_index_mapping


def Implement_EPO(DataPath,iteration_all,batch_size,num_epochs,method_names,x_train_all,c_train_all,x_test_all,c_test_all,arcs,epo_runner):
    cost_EPO = {}
    for iter in iteration_all:
        DataPath_seed = DataPath +"iter="+str(iter)+"/"
        pathlib.Path(DataPath_seed).mkdir(parents=True, exist_ok=True)
        # print("*** seed = ",seed,": Run EPO ******")
        W_EPO,w0_EPO = epo_runner.run(method_names,DataPath_seed,batch_size,num_feat,grid,num_epochs,\
                                        x_train_all[iter],c_train_all[iter],arcs)
        print("w0_EPO=",np.round(w0_EPO,4))
    return cost_EPO


arcs,arc_index_mapping = _getArcs(grid)
num_arcs = len(arcs)
iteration_all = np.arange(0,1)
# obtain data
x_test_all, c_test_all, x_train_all, c_train_all,noise_train_all,noise_test_all,W_star_all = Prepare_Data(DataPath,lower, upper, p, d, coef_seed,iteration_all,num_test, num_train, alpha,mis,data_generation_process,x_dist, e_dist, x_low, x_up, x_mean, x_var, bump)

# EPO, including SPO, PG, LTR
batch_size = 20
num_epochs = 0
method_names = ["spo+","pg","ltr"]
method_names = ["pg"]
from PYEPO import PyEPO_Method
epo_runner = PyEPO_Method()

cost_EPO_all = Implement_EPO(DataPath,[0],batch_size,num_epochs,method_names,x_train_all,c_train_all,x_test_all,c_test_all,arcs,epo_runner)
# print("SPO+Cost = ",np.nanmean(cost_EPO_all[1]["SPO"]))