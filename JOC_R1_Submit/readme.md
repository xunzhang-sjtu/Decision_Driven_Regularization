# This file contains information about DDR

## DDR application
- File: [DDR_All_Results.ipynb](/DDR_Reproduce/DDR_All_Results.ipynb)


## Shortest Path

### Baseline Setting
1. Impact of the Size of Network
    - File: [Baseline_Reproduce_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Reproduce_SPO.ipynb)
    - Figure: [Baseline_Impact_Network_Size.ipynb](/Shortest_Path_Reproduce/Baseline_Impact_Network_Size.ipynb)
    - Data: ".../Data_JOC_R1_Submit/Shortest_Path_Final/Baseline_SPO_Data_Generation/result"

2. Calibrate $\mu$ and $\lambda$: <font color="red">Figure 2</font>
    - File: [Baseline_Calibrate_mu_lambda_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Calibrate_mu_lambda_SPO.ipynb)
    - Figure: [Baseline_Calibrate_mu_lambda_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Calibrate_mu_lambda_SPO.ipynb)
    - Data: ".../Data_JOC_R1_Submit/Shortest_Path_Final/Baseline_SPO_Data_Generation/Result"

2. Comparison with Sequential Learning and Integrated Learning Approaches: <font color="red">Table 1, Figure 3 and 4</font>
    - File: [Baseline_DDR_vs_SLO_and_ILO.ipynb](/Shortest_Path_Reproduce/Baseline_DDR_vs_SLO_and_ILO.ipynb)
    - Figure: [Baseline_DDR_vs_SLO_and_ILO.ipynb](/Shortest_Path_Reproduce/Baseline_DDR_vs_SLO_and_ILO.ipynb)
    - Data: ".../Data_JOC_R1_Submit/Shortest_Path_Final/Baseline_SPO_Data_Generation/Result"


### Various Setting
- Includes: 

    0. Setup parameters: [Various_setting_Reproduce_SPO.ipynb](/Shortest_Path_Reproduce/Various_setting_Reproduce_SPO.ipynb)
    1. Impact of sample size: [Various_Setting_Reproduce_Data_Size_SPO.ipynb](/Shortest_Path_Reproduce/Various_Setting_Reproduce_Data_Size_SPO.ipynb)
    2. Impact of number of feature: [Various_Setting_Reproduce_Num_Feature_SPO.ipynb](/Shortest_Path_Reproduce/Various_Setting_Reproduce_Num_Feature_SPO.ipynb)
    3. Impact of range of error term: [Various_Setting_Reproduce_Noise_Level_SPO.ipynb](/Shortest_Path_Reproduce/Various_Setting_Reproduce_Noise_Level_SPO.ipynb)
    4. Impact of model misspecification: [Various_Setting_Reproduce_Mis_SPO.ipynb](/Shortest_Path_Reproduce/Various_Setting_Reproduce_Mis_SPO.ipynb)
    - Data: ".../Data_JOC_R1_Submit/Shortest_Path_Final/Various_Settings_SPO_Data_Generation/3by3_grid/Result"

### Quadratic term

- File: [Experiment_Model_Quadratic.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Quadratic.ipynb)
- Figure: [Experiment_Model_Quadratic.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Quadratic.ipynb)
- Data: “.../Data_JOC_R1_Submit/Shortest_Path_Final/Quadratic_Term_SPO_Data_Generation/3by3_grid_All/Result”

### Tree_based DDR 

- File: [Experiment_Own.py](Tree_Based_Approaches/Tree_Based_SPO_Plus/Experiment_Own.py)
- Figure: [Figrue_Tree_based_Experiments.ipynb](Tree_Based_Approaches/Tree_Based_SPO_Plus/Figrue_Tree_based_Experiments.ipynb)
- Data: “.../Data_JOC_R1_Submit/Shortest_Path_Tree/dim=3_depth_3_Tree_based_Data_Generation/result/Data_size=200/”




<!-- 
### Model Misspecification
1. Model misspecification when $N=100$: <font color="red">Figure 5, lower left subfig of Figure D.2</font>
    - File: [Experiment_Model_Mis_N_100.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_N_100.ipynb)
    - Figure: [Experiment_Model_Mis_N_100.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_N_100.ipynb)
    - Data: ".../Data_JOC_R1/Shortest_Path_Rep/Model_MisSPO_Data_Generation/3by3_grid/"

1. Model misspecification under different Data Size
    - File: [Experiment_Model_Mis_Data_Size.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_Data_Size.ipynb)
    - Figure: [Experiment_Model_Mis_Data_Size.ipynb](/Shortest_Path_Reproduce/Experiment_Model_Mis_Data_Size.ipynb)
    - Data: “/Data_JOC_R1/Shortest_Path_Rep/Model_Mis_Data_SizeSPO_Data_Generation/3by3_grid/”



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
4. <font color="red">尝试tree-based data generation process</font> -->
