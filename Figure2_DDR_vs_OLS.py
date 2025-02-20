import numpy as np
from numpy import random
import time
import pickle
import os
import pathlib
import inspect

from Data import data_generation
from OLS import ols_method
from DDR import ddr_method
from SPO_Plus import SPO_plus_method
from Performance import performance_evaluation
from Figure import regret_h2h

def store_approach_results(file_store_name,W_results,w0_results,computational_time,y_test,cost):
    dict = {}
    dict["W_results"] = W_results
    dict["w0_results"] = w0_results
    dict["time"] = computational_time
    dict["y_test"] = y_test
    dict["cost"] = cost
    with open(file_store_name, "wb") as tf:
        pickle.dump(dict,tf)

def store_out_of_sample_results(file_name, **kwargs):
    """
    存储变量名和值到字典，并保存为 pickle 文件。

    参数:
    - file_name (str): 要存储的 pickle 文件名
    - **kwargs: 需要存储的变量（自动识别变量名）

    返回:
    - result: 包含变量名和值的字典
    """
    result = {}
    frame = inspect.currentframe().f_back  # 获取上一层调用的栈帧

    # 遍历 kwargs 变量
    for key, value in kwargs.items():
        result[key] = value
        # for var_name, var_value in frame.f_locals.items():
        #     if var_value is value and var_name not in result:  # 避免重复
        #         result[var_name] = var_value

    # 存储为 pickle 文件
    with open(file_name, "wb") as f:
        pickle.dump(result, f)

    # return result

def each_iter_ddr_vs_spo_and_ols(file_path,data, mis, thres, mu, lamb, ypio = 0):
    x_test, z_test_ori, z_test, x_train, z_train_ori, z_train, W_star = data
    perf_eva = performance_evaluation()

    ## Solve and evaluate the OLS model
    ols_method_obj = ols_method()
    W_ols, w0_ols, t_ols, obj_ols = ols_method_obj.ols_solver(file_path,x_train, z_train)
    # print("W_ols = ",W_ols)
    z_test_ols, y_test_ols, c_test_ols = perf_eva.param_prediction_and_cost_estimation(x_test, W_ols, w0_ols, thres)
    c_ols_true =  np.sum(np.minimum(z_test_ori,thres) * y_test_ols, axis = 1)
    
    # print("Average c_ols_true = ",np.nanmean(c_ols_true))

    y_test_opt = perf_eva.decision_finder(z_test_ori)
    c_oracle = np.mean(np.sum(np.minimum(z_test_ori,thres) * y_test_opt, axis = 1))
    
    # print("Average c_oracle = ",np.nanmean(c_oracle))

    ## Solve and evaluate the DDR model
    # Obtain regression parameters
    ddr_method_obj = ddr_method()
    W_ddr, w0_ddr, t_ddr = ddr_method_obj.ddr_solver(x_train, z_train, thres, mu, lamb) #******
    z_test_ddr, y_test_ddr, c_test_ddr = perf_eva.param_prediction_and_cost_estimation(x_test, W_ddr, w0_ddr, thres)
    c_ddr_true =  np.sum(np.minimum(z_test_ori,thres) * y_test_ddr, axis = 1)
    
    # print("Average c_ddr_true = ",np.nanmean(c_ddr_true))


    ## Solve and evaluate the SPO+ model
    # Obtain regression parameters
    spo_method_obj = SPO_plus_method()
    y_train_opt = perf_eva.decision_finder(z_train) #generates the optimal y from the training costs
    W_spo, w0_spo, t_spo = spo_method_obj.spo_solver(x_train, z_train, y_train_opt)
    z_test_spo, y_test_spo, c_test_spo = perf_eva.param_prediction_and_cost_estimation(x_test, W_spo, w0_spo, thres)
    c_spo_true =  np.sum(np.minimum(z_test_ori,thres) * y_test_spo, axis = 1)
    
    # print("Average c_spo_true = ",np.nanmean(c_spo_true))

    store_approach_results(file_path+"oracle.pkl",W_star,W_star,0,y_test_opt,c_oracle)
    store_approach_results(file_path+"OLS.pkl",W_ols,w0_ols,t_ols,y_test_ols,c_ols_true)
    store_approach_results(file_path+"DDR.pkl",W_ddr,w0_ddr,t_ddr,y_test_ddr,c_ddr_true)
    store_approach_results(file_path+"SPO.pkl",W_spo,w0_spo,t_spo,y_test_spo,c_spo_true)
    print("OLS regret = ",np.round(np.nanmean(c_ols_true) - np.nanmean(c_oracle),4),", DDR regret = ",np.round(np.nanmean(c_ddr_true) - np.nanmean(c_oracle),4))

    if ypio == 0:
#     # compares results
        lbels_ddrspo, h2h_ddrspo, mci_ddrspo = perf_eva.cross_compare2(c_ddr_true, c_spo_true, c_oracle)
        lbels_olsspo, h2h_olsspo, mci_olsspo = perf_eva.cross_compare2(c_ols_true, c_spo_true, c_oracle)
        lbels_ddrols, h2h_ddrols, mci_ddrols = perf_eva.cross_compare2(c_ddr_true, c_ols_true, c_oracle)
        return h2h_ddrspo, mci_ddrspo, h2h_olsspo, mci_olsspo, h2h_ddrols, mci_ddrols
    else:
        # compares results plus
        lbels_ddrspo, h2h_ddrspo, mci_ddrspo, pio_ddrspo = perf_eva.cross_compare2plus(c_ddr_true, c_spo_true, c_oracle)
        lbels_olsspo, h2h_olsspo, mci_olsspo, pio_olsspo = perf_eva.cross_compare2plus(c_ols_true, c_spo_true, c_oracle)
        lbels_ddrols, h2h_ddrols, mci_ddrols, pio_ddrols = perf_eva.cross_compare2plus(c_ddr_true, c_ols_true, c_oracle)

    name = file_path+"oos.pkl"
    store_out_of_sample_results(name,\
                                h2h_ddrspo=h2h_ddrspo,mci_ddrspo=mci_ddrspo,pio_ddrspo=pio_ddrspo,\
                                h2h_olsspo=h2h_olsspo,mci_olsspo=mci_olsspo,pio_olsspo=pio_olsspo,\
                                h2h_ddrols=h2h_ddrols,mci_ddrols=mci_ddrols,pio_ddrols=pio_ddrols)

    return h2h_ddrspo, mci_ddrspo, h2h_olsspo, mci_olsspo, h2h_ddrols, mci_ddrols, pio_ddrspo, pio_olsspo, pio_ddrols




if __name__ == "__main__":
    ## Train and test are together
    seed = 3
    start = time.time()
    random.seed(seed)
    iters = 100 
    p = 4
    d = 10
    samples_test = 10000
    samples_train = 100
    lower = 0
    upper = 1
    alpha = 1
    n_epsilon = 1
    mis = 1
    thres = 10000
    ver = 1
    x_dister = 'uniform'
    e_dister = 'normal'
    xl = -2
    xu = 2
    xm = 2
    xv = 0.25
    bp = 7

    mu = 0.25
    lamb = 0.25

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    grandparent_directory = os.path.dirname(parent_directory)
    DataPath = grandparent_directory + '/Data/DDR_vs_OLS/'
    pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)
    print("grandparent_directory:", grandparent_directory)
    print("DataPath:", DataPath)




    Data = {}
    all_h2h_ddrspo = []
    all_mci_ddrspo = []
    all_pio_ddrspo = []

    all_h2h_olsspo = []
    all_mci_olsspo = []
    all_pio_olsspo = []

    all_h2h_ddrols = []
    all_mci_ddrols = []
    all_pio_ddrols = []


    data_gen = data_generation()
    OLS_regret_arr = np.zeros(iters); DDR_regret_arr = np.zeros(iters)
    for i in range(iters):
        print("============== iteration = ",i,"==============")
        file_path = DataPath + "iter="+str(i) +"/"
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        W_star = data_gen.generate_truth(file_path,lower, upper, p, d, version = 0) # o uniform, 1 binary, 2 uniform + feature, 3 binary + feature, 4 sparse, 5 012
        # print("W_star = ",W_star)

        Data[i] = data_gen.generate_samples(file_path,p, d, samples_test, samples_train, alpha, W_star, n_epsilon, mis, thres, 
                            version = ver, x_dist = x_dister, e_dist = e_dister, x_low = xl, x_up = xu, x_mean = xm,
                            x_var = xv, bump = bp)
        # print("Data",Data)

        h2h_ddrspo, mci_ddrspo, h2h_olsspo, mci_olsspo, h2h_ddrols, mci_ddrols, pio_ddrspo, pio_olsspo, pio_ddrols\
        = each_iter_ddr_vs_spo_and_ols(file_path,Data[i], mis, thres, mu, lamb, ypio = 1)
        print("h2h_ddrols = ",h2h_ddrols,"regret reduction = ",pio_ddrols)

        all_h2h_ddrspo.append(h2h_ddrspo*100)
        all_mci_ddrspo.append(mci_ddrspo*100)
        all_pio_ddrspo.append(pio_ddrspo*100)
        
        all_h2h_olsspo.append(h2h_olsspo*100)
        all_mci_olsspo.append(mci_olsspo*100)
        all_pio_olsspo.append(pio_olsspo*100)
        
        all_h2h_ddrols.append(h2h_ddrols*100)
        all_mci_ddrols.append(mci_ddrols*100)
        all_pio_ddrols.append(pio_ddrols*100)
    
    
    store_out_of_sample_results(DataPath+"Result_all.pkl", \
                                all_h2h_ddrspo=all_h2h_ddrspo,all_mci_ddrspo=all_mci_ddrspo,all_pio_ddrspo=all_pio_ddrspo,\
                                all_h2h_olsspo=all_h2h_olsspo,all_mci_olsspo=all_mci_olsspo,all_pio_olsspo=all_pio_olsspo,\
                                all_h2h_ddrols=all_h2h_ddrols,all_mci_ddrols=all_mci_ddrols,all_pio_ddrols=all_pio_ddrols)

    # regret_h2h_fig = regret_h2h()
    # regret_h2h_fig.figure_plot_upleft(all_h2h_ddrols, all_mci_ddrols, figure_name = '411_ddr_ols', size = (5, 5), move = [-0.10, 0.04, 0.30, 0.55])

