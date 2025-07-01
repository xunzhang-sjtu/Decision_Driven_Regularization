# '''
# Runs SPOT (greedy) / CART algorithm on shortest path dataset with nonlinear mapping from features to costs ("nonlinear")
# Outputs algorithm decision costs for each test-set instance as pickle file
# Also outputs optimal decision costs for each test-set instance as pickle file
# Takes multiple input arguments:
#   (1) n_train: number of training observations. can take values 200, 10000
#   (2) eps: parameter (\bar{\epsilon}) in the paper controlling noise in mapping from features to costs.
#     n_train = 200: can take values 0, 0.25
#     n_train = 10000: can take values 0, 0.5
#   (3) deg_set_str: set of deg parameters to try, e.g. "2-10". 
#     deg = parameter "degree" in the paper controlling nonlinearity in mapping from features to costs. 
#     can try values in {2,10}
#   (4) reps_st, reps_end: we provide 10 total datasets corresponding to different generated B values (matrix mapping features to costs). 
#     script will run code on problem instances reps_st to reps_end 
#   (5) max_depth: training depth of tree, e.g. "5"
#   (6) min_weights_per_node: min. number of (weighted) observations per leaf, e.g. "100"
#   (7) algtype: set equal to "MSE" (CART) or "SPO" (SPOT greedy)
# Values of input arguments used in paper:
#   (1) n_train: consider values 200, 10000
#   (2) eps:
#     n_train = 200: considered values 0, 0.25
#     n_train = 10000: considered values 0, 0.5
#   (3) deg_set_str: "2-10"
#   (4) reps_st, reps_end: reps_st = 0, reps_end = 10
#   (5) max_depth: 
#     n_train = 200: considered depths of 1, 2, 3, 1000
#     n_train = 10000: considered depths of 2, 4, 6, 1000
#   (6) min_weights_per_node: 20
#   (7) algtype: "MSE" (CART) or "SPO" (SPOT greedy)
# '''


#data = pickle.load(open('non_linear_big_data_dim4.p','rb'))
#'non_linear_data_dim4.p' has the following options:
#n_train: 200, 400, 800
#nonlinear degrees: 8, 2, 4, 10, 6
#eps: 0, 0.25, 0.5
#50 replications of the experiment (0-49)
#dataset characteristics: 5 continuous features x, dimension 4 grid c, 1000 test set observations
# if n_train == 10000:
#   data = pickle.load(open('non_linear_bigdata10000_dim4.p','rb'))
# else:
#   data = pickle.load(open('non_linear_data_dim4.p','rb'))




import time

import numpy as np
import pickle
from SPO_tree_greedy import SPOTree
from decision_problem_solver import *
import sys
import pathlib
import os
import Oracle


#problem parameters
dim = 3
grid = (dim,dim)
d = (grid[0] - 1) * (grid[0] - 1) * 2 + 2 * (grid[0] - 1) # num of arcs
num_train = 100
num_feat = 5 # size of feature
num_test = 1000
e = 0.5 # scale of normal std or the range of uniform. For the error term
lower = 0 # coef lower bound
upper = 1 # coef upper bound
p = num_feat # num of features
num_nodes = grid[0]*grid[0]
alpha = e # scale of normal std or the range of uniform. For the error term
coef_seed = 1
x_dist = 'uniform'
e_dist = 'normal'
x_low = -2
x_up = 2
x_mean = 2
x_var = 2
bump = 100

deg_set = [1.0]
#evaluate algs of dataset replications from rep_st to rep_end
reps_st = 0 #0 #can be as low as 0
reps_end = 100 #1 #can be as high as 50
iteration_all = np.arange(reps_st,reps_end)

mu_all = np.round(np.arange(0.1,1.0,0.05),4)
lamb_all = np.round(np.arange(0.1,1.0,0.05),4)

valid_frac = 0.0 #set aside valid_frac of training data for validation
########################################
#training parameters
max_depth = 3 #3
min_weights_per_node = 20 #20
algtype = "SPO" #either "MSE" or "SPO", or "DDR"
########################################

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


data_generation_process = "SPO_Data_Generation"
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
DataPath_parent = grandparent_directory + '/Data_JOC_R1/Shortest_Path_Tree/'+str(grid[0])+'by'+str(grid[1])+'_grid' +'_depth_'+str(max_depth)+"_0629/"
pathlib.Path(DataPath_parent).mkdir(parents=True, exist_ok=True)
print("DataPath_parent:", DataPath_parent)
result_dir = DataPath_parent +"result/Data_size="+str(num_train)+"/"
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
print("result_dir:", result_dir)


print(np.arange(reps_st,reps_end))
# #costs_deg[deg] yields a n_reps*n_test matrix of costs corresponding to the experimental data for deg, i.e.
# #costs_deg[deg][i][j] gives the observed cost on test set i (0-49) example j (0-(n_test-1))
costs_deg = {}; cost_MSE = {}; cost_SPO = {}; cost_DDR = {}; cost_Oracle = {}
optcosts_deg = {} #optimal costs
for deg in deg_set:
    costs_deg[deg] = np.zeros((len(iteration_all),num_test))
    optcosts_deg[deg] = np.zeros((len(iteration_all),num_test))

    DataPath = DataPath_parent + "data_size="+str(num_train)+"_deg="+str(deg)+"_e="+str(e)+"_d="+str(d)+"_p="+str(p)+"_x_dist="+x_dist+"_num_test="+str(num_test)+"/"
    pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)

    x_test_all, c_test_all, x_train_all, c_train_all, noise_train_all,noise_test_all,W_star_all = \
    Prepare_Data(DataPath,lower, upper, p, d, coef_seed,iteration_all,num_test, num_train, alpha,deg,data_generation_process,x_dist, e_dist, x_low, x_up, x_mean, x_var, bump)

    for iter in iteration_all:
        train_x = x_train_all[iter]
        train_cost = c_train_all[iter]
        test_x = x_test_all[iter]
        test_cost = c_test_all[iter]
        print("Deg "+str(deg)+", iter "+str(iter))

        #split up training data into train/valid split
        n_valid = int(np.floor(num_train*valid_frac))
        valid_x = train_x[:n_valid]
        valid_cost = train_cost[:n_valid]
        train_x = train_x[n_valid:]
        train_cost = train_cost[n_valid:]

        ### Run Oracle ###
        cost_Oracle[deg,iter],cost_oracle_pred = Oracle.compute_out_of_sample_cost(deg,W_star_all[iter],x_test_all[iter],c_test_all[iter],num_feat,dim)
        print("iter = ",iter, ",Oracle avg cost = ",np.mean(cost_Oracle[deg,iter]))

        # ### Run MSE Tree ###
        SPO_weight_param = 0.0
        MSE_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
        MSE_tree.fit(0,0,dim,train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
        pred_decision = MSE_tree.est_decision(test_x)
        cost_MSE[deg,iter] = [np.sum(cost_oracle_pred[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
        print("iter = ",iter, ",MSE avg cost = ",np.mean(cost_MSE[deg,iter]))

        ### Run SPO Tree ###
        SPO_weight_param = 1.0
        SPO_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
        SPO_tree.fit(0,0,dim,train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
        pred_decision = SPO_tree.est_decision(test_x)
        cost_SPO[deg,iter] = [np.sum(cost_oracle_pred[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
        print("iter = ",iter, ",SPO avg cost = ",np.mean(cost_SPO[deg,iter]))

        ### Run DDR Tree ###
        SPO_weight_param = 2.0
        for mu in mu_all:
            for lamb in lamb_all:
                DDR_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
                DDR_tree.fit(mu,lamb,dim,train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
                pred_decision = DDR_tree.est_decision(test_x)
                cost_DDR[mu,lamb,deg,iter] = [np.sum(cost_oracle_pred[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
                # print("iter = ",iter, ",mu=",mu,",lamb=",lamb,",DDR avg cost = ",np.mean(cost_DDR[mu,lamb,deg,iter]))
                print("iter = ",iter, ",mu=",mu,",lamb=",lamb,\
                    ",DDR vs SPO cost = ",np.round(np.mean(cost_DDR[mu,lamb,deg,iter])/np.mean(cost_SPO[deg,iter]),4),\
                    ",DDR vs MSE cost = ",np.round(np.mean(cost_DDR[mu,lamb,deg,iter])/np.mean(cost_MSE[deg,iter]),4))
            print("-----------------------------")

        result_dir_deg = result_dir + "deg="+str(deg)+"/"
        print("result_dir_deg:", result_dir_deg)
        pathlib.Path(result_dir_deg).mkdir(parents=True, exist_ok=True)
        with open(result_dir_deg+'cost_Oracle.pkl', "wb") as tf:
            pickle.dump(cost_Oracle,tf)
        with open(result_dir_deg+'cost_MSE.pkl', "wb") as tf:
            pickle.dump(cost_MSE,tf)
        with open(result_dir_deg+'cost_SPO.pkl', "wb") as tf:
            pickle.dump(cost_SPO,tf)
        with open(result_dir_deg+'cost_DDR.pkl', "wb") as tf:
            pickle.dump(cost_DDR,tf)