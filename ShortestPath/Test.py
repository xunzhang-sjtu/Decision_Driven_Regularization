import random
import numpy as np
import torch
import os
import pathlib
import pickle

# data_generation_process = "SPO_Data_Generation"
data_generation_process = "DDR_Generation"

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
num_data = 100 # number of training data
num_feat = 5 # size of feature
num_test = 1
deg = 1.0 # polynomial degree
e = 1.0 # noise width

DataPath = DataPath + "data_size="+str(num_data)+"_deg="+str(deg)+"_e="+str(e)+"/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)


def obtain_data(data_generation_process,num_data,num_test, num_feat, grid, deg, e, seed):
    from Data import data_generation
    data_gen = data_generation()
    if data_generation_process == "SPO_Data_Generation":
        feats, costs = data_gen.generate_Shortest_Path_Data(num_data+num_test, num_feat, grid, deg, e, seed=seed)
        # split train test data
        from sklearn.model_selection import train_test_split
        x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=num_test, random_state=42)

    if data_generation_process == "DDR_Generation":
        lower = 0
        upper = 1
        p = 5
        d = 40
        alpha = 1
        mis = deg
        n_epsilon = 1
        W_star = data_gen.generate_truth("",lower, upper, p, d, seed,version = 0) 
        # print("W_star = ",W_star[0,:])
        x_test, z_test_ori, c_test, x_train, z_train_ori, c_train, W_star = data_gen.generate_samples("",p, d, num_test, num_data, alpha, W_star, n_epsilon, mis, thres = 10, 
                                version = 1, x_dist = 'normal', e_dist = 'normal', x_low = 0, x_up = 2, x_mean = 2, x_var = 0.25, bump = 0) 

    return x_train, x_test, c_train, c_test



seed_all = np.arange(1,2)
cost_Oracle_all = {}; cost_SPO_all = {}; cost_OLS_all = {}; cost_DDR_all = {}

for seed in seed_all:
    DataPath_seed = DataPath +"Seed="+str(seed)+"/"
    pathlib.Path(DataPath_seed).mkdir(parents=True, exist_ok=True)

    # #  ****** Data generation *********
    x_train, x_test, c_train, c_test = obtain_data(data_generation_process,num_data,num_test, num_feat, grid, deg, e, seed)

    raw_data = {}
    raw_data["x_train"] = x_train; raw_data["x_test"] = x_test; raw_data["c_train"] = c_train; raw_data["c_test"] = c_test
    with open(DataPath_seed +'raw_data.pkl', "wb") as tf:
        pickle.dump(raw_data,tf)

    #  ****** SPO *********
    print("*** seed = ",seed,": Run SPO ========")
    from SPO_Plus import run_SPO_Shortest_Path
    SPO_runner = run_SPO_Shortest_Path()
    batch_size = 20
    num_epochs = 1
    arcs,loader_train,loader_test,cost_Oracle_all[seed],cost_SPO_all[seed] = SPO_runner.run(DataPath_seed,x_train,c_train,x_test,c_test,batch_size,num_feat,grid,num_epochs,True)
    print("Average Oracle Cost = ",np.mean(cost_Oracle_all[seed]))
    print("Average SPO Cost = ",np.mean(cost_SPO_all[seed]))

    #  ****** OLS *********
    print("*** seed = ",seed,": Run OLS ========")
    from OLS import run_OLS_Shortest_Path
    OLS_runner = run_OLS_Shortest_Path()
    cost_OLS_all[seed] = OLS_runner.run(DataPath_seed,arcs,x_train,c_train,grid,loader_test,loader_train)
    print("Average OLS Cost = ",np.mean(cost_OLS_all[seed]))


    # #  ****** DDR *********
    print("*** seed = ",seed,": Run DDR ========")
    from DDR import run_DDR_Shortest_Path
    DDR_runner = run_DDR_Shortest_Path()
    mu_arr = np.arange(-0.5,0.5,0.05)
    lamb_arr = np.arange(0.1,1,0.05)
    # lamb_arr = [0.05,0.1,0.15,0.2]
    minimum_value = 1000000000

    cost_DDR_all[seed] = DDR_runner.run(DataPath_seed,lamb_arr,mu_arr,arcs,x_train, c_train, grid,loader_test,num_nodes=25)
