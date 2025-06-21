# This file contains information about DDR

## DDR application
- File: [DDR_All_Results.ipynb](/DDR_Reproduce/DDR_All_Results.ipynb)


## Shortest Path

### Baseline Setting
- Includes: 
    1. DDR vs OLS; 
    2. Impact of the size of Network
- File: [Baseline_Reproduce_SPO.ipynb](/Shortest_Path_Reproduce/Baseline_Reproduce_SPO.ipynb)
- <font color="red">Run N = 300, 检查是否能保证和N=100下类似的结果
```
middle_path = '/Data_JOC_R1/Shortest_Path_0620/Baseline_' + data_generation_process +"/"
```
</font>



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