import numpy as np
from Performance import performance_evaluation
perfs = performance_evaluation()

class Oracle_Processing:
    def __init__(self):
        pass

    def Implement_Oracle(self,arcs, grid,mis,bump,W_star_all,x_test_all,noise_test_all,iteration_all,num_feat,data_generation_process):
        cost_Oracle_Post = {}; cost_Oracle_Ante = {}
        for iter in iteration_all:
            if data_generation_process == "SPO_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                # cost_Oracle_Post[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Post(arcs, grid,cost_oracle_pred,cost_oracle_pred,noise_test_all[iter])
                cost_Oracle_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_oracle_pred,cost_oracle_pred)

            if data_generation_process == "DDR_Data_Generation":
                cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                cost_oracle_pred = (cost_oracle_ori ** mis).T
                cost_Oracle_Ante[iter] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_oracle_pred,cost_oracle_pred)
            if iter % 20 == 0 and iter > 0:
                # print("Oracle: iter=",iter,",cost_Oracle_Post=",np.nanmean(cost_Oracle_Post[iter]),",cost_Oracle_Ante=",np.nanmean(cost_Oracle_Ante[iter]))
                print("Oracle: iter=",iter,",cost_Oracle_Ante=",np.nanmean(cost_Oracle_Ante[iter]))
        return cost_Oracle_Post,cost_Oracle_Ante