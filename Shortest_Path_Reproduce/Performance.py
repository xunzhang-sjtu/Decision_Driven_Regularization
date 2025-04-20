import numpy as np
import time
from gurobipy import *

import torch

class performance_evaluation:
    def __init__(self):
        pass

    # from pyepo import EPO
    def compute_EPO_Cost(self,full_shortest_model,predmodel, dataloader,arcs,grid):
        # evaluate
        predmodel.eval()
        cost_pred_arr = []
        # load data
        for data in dataloader:
            x, c, w, z = data
            # cuda
            if next(predmodel.parameters()).is_cuda:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # predict
            with torch.no_grad(): # no grad
                cp = predmodel(x).to("cpu").detach().numpy()
            # print("cp[0] = ",cp[0])
            # solve
            for j in range(cp.shape[0]):
                sol_pred = full_shortest_model.solve_Shortest_Path(arcs,cp[j],grid)
                cost_pred = np.dot(sol_pred, c[j].to("cpu").detach().numpy())
                cost_pred_arr.append(cost_pred)
                # cost_true_arr.append(z[j].item())
        # turn back train mode
        predmodel.train()
        # normalized
        return cost_pred_arr
    
    # def compute_DDR_out_of_sample_Cost(self,arcs, grid,c_pred,c_oracle_avg,noise,with_noise):
    #     from Shortest_Path_Model import My_ShortestPathModel
    #     full_shortest_model = My_ShortestPathModel()        
    #     cost_pred_arr = []
    #     if with_noise:
    #         for j in range(np.shape(c_pred)[0]):
    #             sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
    #             c_oralce_realization = c_oracle_avg[j,:] + noise
    #             cost_real = np.nanmean(c_oralce_realization @ sol_pred)
    #             cost_pred_arr.append(cost_real)
    #     else:
    #         for j in range(np.shape(c_pred)[0]):
    #             sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
    #             cost_real = c_oracle_avg[j,:] @ sol_pred
    #             cost_pred_arr.append(cost_real)

    #     return np.asarray(cost_pred_arr)
    
    def compute_out_of_sample_Cost(self,arcs, grid,c_pred,noise,mis,W_star,bump,x_test,num_feat,data_generation_process):

        if data_generation_process == "SPO_Data_Generation":
            cost_oracle_ori = (W_star @ x_test.T)/np.sqrt(num_feat) + 3
            cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
            cost_output = self.compute_SPO_out_of_sample_Cost(arcs, grid,c_pred,cost_oracle_pred,noise)

        if data_generation_process == "DDR_Data_Generation":
            cost_oracle_ori = (W_star @ x_test.T) + bump
            cost_oracle_pred = (cost_oracle_ori ** mis).T
            # cost_DDR_with_noise_all[iter,mu,lamb] = perfs.compute_DDR_out_of_sample_Cost(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter],True)
            cost_output = self.compute_DDR_out_of_sample_Cost(arcs, grid,c_pred,cost_oracle_pred,noise,False)
    
        return np.asarray(cost_output)

    def compute_SPO_out_of_sample_Cost(self,arcs, grid,c_pred,c_oracle_avg,noise):
        from Shortest_Path_Model import My_ShortestPathModel
        full_shortest_model = My_ShortestPathModel()
        # evaluate
        cost_pred_arr = []
        for j in range(np.shape(c_pred)[0]):
            sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
            c_oralce_realization = c_oracle_avg[j,:] * noise
            cost_real = np.nanmean(c_oralce_realization @ sol_pred)
            cost_pred_arr.append(cost_real)

        return np.asarray(cost_pred_arr)

    # def cross_compare2plus(self,c_item, c_base, c_oracle):
    #     N = len(c_item)
    #     c_diff = c_base - c_item
    #     lbel = np.zeros((N,1))
        
    #     equals = np.sum(c_diff == 0)
    #     wins = np.sum(c_diff > 0) # indicate num of c_item is lower than c_base
    #     lose = np.sum(c_diff < 0)
        
    #     lbel[c_diff < 0] = 1
    #     lbel[c_diff > 0] = -1
        
    # #     print(N, equals, wins, lose)
    #     if N == equals:
    #         win_ratio = 0.5
    #     else:
    #         win_ratio = wins/(N - equals)
    #     cost_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base))
    #     regret_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base) - np.mean(c_oracle))
    #     return win_ratio, cost_reduction, regret_reduction
    
