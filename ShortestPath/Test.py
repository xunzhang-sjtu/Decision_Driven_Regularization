import random
import numpy as np
import torch
import os
import pathlib
import pickle

data_generation_process = "SPO_Data_Generation"
# data_generation_process = "DDR_Generation"

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
DataPath = os.path.dirname(grandparent_directory) + '/Data/' + data_generation_process + "/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)
print("grandparent_directory:", grandparent_directory)
print("DataPath:", DataPath)


# import pyepo
# generate data
grid = (5,5) # grid size
num_train = 100 # number of training data
num_feat = 5 # size of feature
num_test = 1000
deg = 4.0 # polynomial degree
e = 0.5 # noise width

lower = 0 # coef lower bound
upper = 1 # coef upper bound
p = 5 # num of features
d = 40 # num of arcs
alpha = 1.0 # scale of normal std or the range of uniform. For the error term
mis = deg # model misspecification

DataPath = DataPath + "data_size="+str(num_train)+"_deg="+str(deg)+"_e="+str(e)+"/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)

def Prepare_Data(DataPath,lower, upper, p, d, coef_seed,seed_all,num_test, num_train, alpha,mis,Data_generation_Version):
# #  ****** Coef generation *********
    from Data import data_generation
    data_gen = data_generation()
    # print("W_star = ",W_star[0,:])
    W_star = data_gen.generate_truth("",lower, upper, p, d, coef_seed,version = 0) 

    for seed in seed_all:
        DataPath_seed = DataPath +"Seed="+str(seed)+"/"
        pathlib.Path(DataPath_seed).mkdir(parents=True, exist_ok=True)
        # #  ****** Data generation *********
        x_test, c_test, x_train, c_train, W_star = data_gen.generate_samples(seed,DataPath_seed,p, d, num_test, num_train, alpha, W_star, mis, thres = 10, 
                                version = "SPO_Data_Generation", x_dist = 'normal', e_dist = 'normal', x_low = 0, x_up = 2, x_mean = 2, x_var = 0.25, bump = 0) 
        print()

coef_seed = 2
seed_all = np.arange(1,3)
Prepare_Data(DataPath,lower, upper, p, d, coef_seed,seed_all,num_test, num_train, alpha,mis,data_generation_process)


from SPO_Plus import shortestPathModel
SPM = shortestPathModel()
arcs = SPM._getArcs()


cost_Oracle_all = {}
for seed in seed_all:
    DataPath_seed = DataPath +"Seed="+str(seed)+"/"
    pathlib.Path(DataPath_seed).mkdir(parents=True, exist_ok=True)
    with open(DataPath_seed+'Data.pkl', "rb") as tf:
        Data = pickle.load(tf)
    c_test = Data["c_test"]
    from Peformance import performance_evaluation
    perfs = performance_evaluation()
    cost_Oracle_all[seed] = perfs.compute_Oracel_Cost(arcs, grid,c_test)
    print("*** seed = ",seed," Average Oracle Cost = ",np.mean(cost_Oracle_all[seed]))

from PYEPO import PyEPO_Method
epo_runner = PyEPO_Method()

method_names = ["spo+","pg"]
for seed in seed_all:
    DataPath_seed = DataPath +"Seed="+str(seed)+"/"
    pathlib.Path(DataPath_seed).mkdir(parents=True, exist_ok=True)
    batch_size = 20
    num_epochs = 30
    epo_runner.run(method_names,DataPath_seed,batch_size,num_feat,grid,num_epochs)
