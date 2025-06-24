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
- Includes: 
    1. DDR vs OLS,SPO+,PG,LTR when N = 100
    2. DDR vs OLS,SPO+,PG,LTR when N = 500 <font color="red">：MacMini 上运行，已经运行完了，需要绘制h2h and regret reduction distribution change，6.21晚 </font>


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