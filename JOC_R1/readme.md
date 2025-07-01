# This file contains information about DDR

## DDR application
- File: [DDR_All_Results.ipynb](/DDR_Reproduce/DDR_All_Results.ipynb)


## Shortest Path


### Baseline Setting
1. Impact of the Size of Network
    - File: [Baseline_Reproduce_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Reproduce_SPO.ipynb)
    - Figure: [Baseline_Impact_Network_Size.ipynb](/Shortest_Path_Reproduce/Baseline_Impact_Network_Size.ipynb)
    - Data: ".../Shortest_Path_Final"

2. Calibrate $\mu$ and $\lambda$
    - File: [Baseline_Calibrate_mu_lambda_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Calibrate_mu_lambda_SPO.ipynb)
    - Figure: [Baseline_Calibrate_mu_lambda_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Calibrate_mu_lambda_SPO.ipynb)
    - Data: ".../Shortest_Path_Final"

2. Comparison with Sequential Learning and Integrated Learning Approaches
    - File: [Baseline_DDR_vs_SLO_and_ILO.ipynb](/Shortest_Path_Reproduce/Baseline_DDR_vs_SLO_and_ILO.ipynb)
    - Figure: [Baseline_DDR_vs_SLO_and_ILO.ipynb](/Shortest_Path_Reproduce/Baseline_DDR_vs_SLO_and_ILO.ipynb)
    - Data: ".../Shortest_Path_Final"


### Model Misspecification
1. Model misspecification when $N=100$
    - File: [Experiment_Model_Mis_N_100.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_N_100.ipynb)
    - Figure: [Experiment_Model_Mis_N_100.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_N_100.ipynb)
    - Data: ".../Data_JOC_R1/Shortest_Path_Rep/Model_MisSPO_Data_Generation/3by3_grid/"

1. Model misspecification under different Data Size
    - File: [Experiment_Model_Mis_Data_Size.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_Data_Size.ipynb)
    - Figure: [Experiment_Model_Mis_Data_Size.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_Data_Size.ipynb)
    - Data: “/Data_JOC_R1/Shortest_Path_Rep/Model_Mis_Data_SizeSPO_Data_Generation/3by3_grid/”


### Quadratic term
- File: [Baseline_Quadratic.ipynb](/Shortest_Path_Reproduce/Baseline_Quadratic.ipynb)

- <font color="blue">：Baseline setting 上考虑所有的quadratic term, 包括自身和交叉项，结果不如不考虑quadratic term </font>
- 只考虑自身的quadratic term <font color="red">：中断了该实验，因为要去做MIP的实验，6.23 上午9点</font>
- 增加sample size 数量 <font color="red">：tbd </font>
- 增加model misspecification <font color="red">：tbd </font>


### Various Setting
- Includes: 
    1. Impact of sample size; 
    2. Impact of number of feature
    3. Impact of range of error term
    4. Impact of model misspecification
- File: [Various_setting_Reproduce_SPO.ipynb](/Shortest_Path_Reproduce/Various_setting_Reproduce_SPO.ipynb)

- 复现时，需要在Data_Simulator函数中，使得data_generation_process前面的SEED为iter，如下 W_star = data_gen.generate_truth(DataPath_iter,lower, upper, p, d, iter,data_generation_process)

### Revise codes
- run EPO时，我在 dataset.py 中关闭了Optimizing for optDataset... 和tqdm的打印


### DDR Tree
1. 在mtp.py 文件中增加,计算root node 上DDR cost，需要我们也对error()函数内做了修改。
```
if leaf_mod.SPO_weight_param == 2.0:
    DDR_loss_Rst = leaf_mod.error(A,Y,mu,lamb)
    leaf_mod_error = DDR_loss_Rst["obj"]
else:
    leaf_mod_error = fast_avg(leaf_mod.error(A,Y,mu,lamb),weights)
```

2. mtp.py _find_best_split_binary()函数中，做如下代码替换：
```
l_avg_errors[k] = np.dot(leaf_mod_l.error(A_l,Y_l),weights_train_l)/sum_weights;
r_avg_errors[k] = np.dot(leaf_mod_r.error(A_r,Y_r),weights_train_r)/sum_weights;
```
替换为：
```
if leaf_mod_l.SPO_weight_param == 2.0:
    DDR_loss_Rst = leaf_mod_l.error(A_l,Y_l,tree_params.mu,tree_params.lamb)
    l_avg_errors[k] = DDR_loss_Rst["obj"]
else:
    l_avg_errors[k] = np.dot(leaf_mod_l.error(A_l,Y_l,tree_params.mu,tree_params.lamb),weights_train_l)/sum_weights;

if leaf_mod_r.SPO_weight_param == 2.0:
    DDR_loss_Rst = leaf_mod_r.error(A_r,Y_r,tree_params.mu,tree_params.lamb)
    r_avg_errors[k] = DDR_loss_Rst["obj"]
else:
    r_avg_errors[k] = np.dot(leaf_mod_r.error(A_r,Y_r,tree_params.mu,tree_params.lamb),weights_train_r)/sum_weights;
```
<font color="red">Experimental Records</font>
1. S = 100, 2*2 grid, max depth = 2, Not good as the SPO
2. S = 100, 2*2 grid, max depth = 3, Try 
3. S = 200, 2*2 grid, max depth = 3, p = 1, deg = 1.0, lambda = 500, mu = np.round(np.arange(0.85,1.0,0.025),4), 
    - 可以找到比SPO好，但没有MSE好，
    - '/Data_JOC_R1/Shortest_Path_Tree/'+str(grid[0])+'by'+str(grid[1])+'_grid' +'_depth_'+str(max_depth)+"_0628/"