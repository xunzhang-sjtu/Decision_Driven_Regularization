import time

import numpy as np
import pickle
from SPO_tree_greedy import SPOTree
from decision_problem_solver import *
import sys
import pathlib
import os
import Oracle
np.random.seed(1)

def generate_samples(file_path,p,d,num_test, num_train, alpha, e_dist, tree,feat_names,seed):

    # if os.path.exists(file_path+'Data.pkl'):
    if False:

        with open(file_path+'Data.pkl', "rb") as tf:
            Data = pickle.load(tf)

        x_train = Data["x_train"]
        c_train_avg = Data["c_train_avg"]
        c_train_real = Data["c_train_real"]
        x_test = Data["x_test"]
        c_test_avg = Data["c_test_avg"]
        c_test_real = Data["c_test_real"]
    else:
        # np.random.seed(seed)
        n = num_train+num_test
        c_avg = np.zeros((n, d))
        c_real = np.zeros((n, d))
        x = np.zeros((n,p))
        eps_all = np.zeros((n,d))
        if e_dist == 'normal':
            eps_all = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = n)
        elif e_dist == 'uniform':
            eps_all = np.random.uniform(-alpha,alpha, size = (n, d))

        for i in range(n):
            sample = {}
            f_index = 0
            for feat_name in feat_names:
                x[i,f_index] = np.random.uniform(-1, 1)
                sample[feat_name] = x[i,f_index]
                f_index = f_index + 1
            c_avg[i,:] = tree.evaluate(sample)
            c_real[i,:] = c_avg[i,:] + eps_all[i,:]

        x_train = x[0:num_train,:]
        c_train_avg = c_avg[0:num_train,:]
        c_train_real = c_real[0:num_train,:]
        x_test = x[num_train:n,:]
        c_test_avg = c_avg[num_train:n,:]
        c_test_real = c_real[num_train:n,:]

        dict = {}
        dict["x_train"] = x_train
        dict["c_train_avg"] = c_train_avg
        dict["c_train_real"] = c_train_real
        dict["x_test"] = x_test
        dict["c_test_avg"] = c_test_avg
        dict["c_test_real"] = c_test_real
        with open(file_path+'Data.pkl', "wb") as tf:
            pickle.dump(dict,tf)
    return x_train, c_train_avg, c_train_real, x_test, c_test_avg,c_test_real

#problem parameters
dim = 3
grid = (dim,dim)
d = (grid[0] - 1) * (grid[0] - 1) * 2 + 2 * (grid[0] - 1) # num of arcs
num_train = 100
num_feat = 5 # size of feature
num_test = 1000
e = 10 # scale of normal std or the range of uniform. For the error term
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
reps_end = 2 #1 #can be as high as 50
iteration_all = np.arange(reps_st,reps_end)

mu_all = np.round(np.arange(0.1,1.0,0.1),4)
mu_all = [0.5]
lamb_all = np.append(np.append(np.round(np.arange(0.1,1.0,0.2),4),np.arange(1.0,10.0,2.0)),np.arange(10,100,20))
lamb_all = np.arange(1000,10000,1000)

valid_frac = 0.0 #set aside valid_frac of training data for validation
########################################
#training parameters
max_depth = 3 #3
min_weights_per_node = 10 #20
algtype = "SPO" #either "MSE" or "SPO", or "DDR"
########################################


data_generation_process = "Tree_based_Data_Generation"
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
DataPath_parent = grandparent_directory + '/Data_JOC_R1/Shortest_Path_Tree/'+str(grid[0])+'by'+str(grid[1])+'_grid' +'_depth_'+str(max_depth)+"_Tree_based_Data_Generation/"
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


    for iter in iteration_all:
        from Data import VectorBinaryTreeNode
        feat_names = [f"x_{i}" for i in range(num_feat)]
        tree = VectorBinaryTreeNode(depth=max_depth, max_depth=max_depth, output_dim=d, current_depth=0, feature_names=feat_names)

        # # 2. 打印所有叶子节点的规则和向量值
        # print("*********************************")
        # print("二叉树所有路径规则和叶子节点向量值:")
        # for i, rule_info in enumerate(tree.get_rules()):
        #     print(f"路径 {i+1}: {' AND '.join(rule_info['rules'])}")
        #     print(f"向量值: {rule_info['value']}")
        #     print("-" * 50)

        x_train, c_train_ante, c_train_post, x_test, c_test_ante, c_test_post = generate_samples(DataPath,p,d,num_test, num_train, alpha, e_dist, tree,feat_names,1)
        # print("x_train = ",x_train[0,:])
        ### Run Oracle ###
        cost_Oracle[deg,iter] = Oracle.compute_out_of_sample_cost_tree(c_test_ante,dim)
        print("iter = ",iter, ",Oracle avg cost = ",np.mean(cost_Oracle[deg,iter]))

        # ### Run MSE Tree ###
        SPO_weight_param = 0.0
        MSE_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
        MSE_tree.fit(0,0,dim,x_train,c_train_post,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
        pred_decision = MSE_tree.est_decision(x_test)
        cost_MSE[deg,iter] = [np.sum(c_test_ante[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
        print("iter = ",iter, ",MSE avg cost = ",np.mean(cost_MSE[deg,iter]))

        ### Run SPO Tree ###
        SPO_weight_param = 1.0
        SPO_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
        SPO_tree.fit(0,0,dim,x_train,c_train_post,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
        pred_decision = SPO_tree.est_decision(x_test)
        cost_SPO[deg,iter] = [np.sum(c_test_ante[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
        print("iter = ",iter, ",SPO avg cost = ",np.mean(cost_SPO[deg,iter]))

        ### Run DDR Tree ###
        SPO_weight_param = 2.0
        for mu in mu_all:
            for lamb in lamb_all:
                DDR_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
                DDR_tree.fit(mu,lamb,dim,x_train,c_train_post,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
                pred_decision = DDR_tree.est_decision(x_test)
                cost_DDR[mu,lamb,deg,iter] = [np.sum(c_test_ante[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
                # print("iter = ",iter, ",mu=",mu,",lamb=",lamb,",DDR avg cost = ",np.mean(cost_DDR[mu,lamb,deg,iter]))
                print("iter = ",iter, ",mu=",mu,",lamb=",lamb,\
                    ",DDR vs SPO cost = ",np.round(np.mean(cost_DDR[mu,lamb,deg,iter])/np.mean(cost_SPO[deg,iter]),4),\
                    ",DDR vs MSE cost = ",np.round(np.mean(cost_DDR[mu,lamb,deg,iter])/np.mean(cost_MSE[deg,iter]),4))
            print("-----------------------------")

    #     result_dir_deg = result_dir + "deg="+str(deg)+"/"
    #     print("result_dir_deg:", result_dir_deg)
    #     pathlib.Path(result_dir_deg).mkdir(parents=True, exist_ok=True)
    #     with open(result_dir_deg+'cost_Oracle.pkl', "wb") as tf:
    #         pickle.dump(cost_Oracle,tf)
    #     with open(result_dir_deg+'cost_MSE.pkl', "wb") as tf:
    #         pickle.dump(cost_MSE,tf)
    #     with open(result_dir_deg+'cost_SPO.pkl', "wb") as tf:
    #         pickle.dump(cost_SPO,tf)
    #     with open(result_dir_deg+'cost_DDR.pkl', "wb") as tf:
    #         pickle.dump(cost_DDR,tf)