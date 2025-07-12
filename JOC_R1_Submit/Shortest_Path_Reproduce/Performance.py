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

    def compute_SPO_out_of_sample_Cost_Ex_Post(self,arcs, grid,c_pred,c_oracle_avg,noise):
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

    def compute_SPO_out_of_sample_Cost_Ex_Ante(self,arcs, grid,c_pred,c_oracle_avg):
        from Shortest_Path_Model import My_ShortestPathModel
        full_shortest_model = My_ShortestPathModel()
        # evaluate
        cost_pred_arr = []
        for j in range(np.shape(c_pred)[0]):
            sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
            cost_real = c_oracle_avg[j,:] @ sol_pred
            cost_pred_arr.append(cost_real)
        return np.asarray(cost_pred_arr)

class H2h_Regret_Evaluation:
    def __init__(self):
        pass

    def cross_compare2plus(self,c_item, c_base, c_oracle):
        N = len(c_item)
        c_diff = c_base - c_item
        lbel = np.zeros((N,1))
        
        equals = np.sum(c_diff == 0)
        wins = np.sum(c_diff > 0) # indicate num of c_item is lower than c_base
        lose = np.sum(c_diff < 0)
        
        lbel[c_diff < 0] = 1
        lbel[c_diff > 0] = -1
        
    #     print(N, equals, wins, lose)
        if N == equals:
            win_ratio = 0.5
        else:
            win_ratio = wins/(N - equals)
        cost_reduction = (np.nanmean(c_diff))/np.abs(np.nanmean(c_base))
        if np.nanmean(c_base) - np.nanmean(c_oracle) <= 1e-6:
            regret_reduction = 0.0
        else:
            regret_reduction = (np.nanmean(c_diff))/np.abs(np.nanmean(c_base) - np.nanmean(c_oracle))
        return win_ratio, cost_reduction, regret_reduction
    
    def calculate_h2h_regret(self,mu,lamb,iteration_all,cost_DDR_Post_all,cost_OLS_Post_all,cost_Oracle_Post_all,cost_DDR_Ante_all,cost_OLS_Ante_all,cost_Oracle_Ante_all):
        ### Post Result
        h2h_ = np.zeros(len(iteration_all)); cost_rd_ = np.zeros(len(iteration_all)); regret_rd_ = np.zeros(len(iteration_all))
        # for iter_index in range(len(iteration_all)):
        #     iter = iteration_all[iter_index]
        #     h2h_[iter_index],cost_rd_[iter_index],regret_rd_[iter_index] = cross_compare2plus(cost_DDR_Post_all[iter,mu,lamb], cost_OLS_Post_all[iter], cost_Oracle_Post_all[iter])
        # regret_post = np.round( len(regret_rd_[regret_rd_ > 0.0])/len(regret_rd_),4 )
        # h2h_post = np.round( len(h2h_[h2h_ >= 0.5])/len(h2h_),4 )

        ### Ante Result
        h2h_ = np.zeros(len(iteration_all)); cost_rd_ = np.zeros(len(iteration_all)); regret_rd_ = np.zeros(len(iteration_all))
        for iter_index in range(len(iteration_all)):
            iter = iteration_all[iter_index]
            h2h_[iter_index],cost_rd_[iter_index],regret_rd_[iter_index] = self.cross_compare2plus(cost_DDR_Ante_all[iter,mu,lamb], cost_OLS_Ante_all[iter], cost_Oracle_Ante_all[iter])
        regret_ante = np.round( len(regret_rd_[regret_rd_ > 0.0])/len(regret_rd_),4 )
        h2h_ante = np.round( len(h2h_[h2h_ >= 0.5])/len(h2h_),4 )
        # return h2h_post,regret_post,h2h_ante,regret_ante
        return h2h_ante,regret_ante
    
    def calculate_DDR_vs_Others_h2h_regret(self,mu,lamb,iteration_all,cost_DDR_Ante_all,cost_other_all,cost_Oracle_Ante_all):
        
        h2h_ = np.zeros(len(iteration_all)); cost_rd_ = np.zeros(len(iteration_all)); regret_rd_ = np.zeros(len(iteration_all))
        for iter_index in range(len(iteration_all)):
            iter = iteration_all[iter_index]
            h2h_[iter_index],cost_rd_[iter_index],regret_rd_[iter_index] = self.cross_compare2plus(cost_DDR_Ante_all[iter,mu,lamb], cost_other_all[iter], cost_Oracle_Ante_all[iter])

        # return h2h_post,regret_post,h2h_ante,regret_ante
        return h2h_,regret_rd_



    def compute_h2h_regret_DDR_vs_OLS_diff_setting(self,Data_LSM,mu,lamb,iteration_all,params_all,num_train,deg,e,d,p,x_dist,num_test,DataPath_Parent,which_param):
        regret_ = np.zeros(len(params_all)); h2h_ = np.zeros(len(params_all))

        for _index in range(len(params_all)):
            param = params_all[_index]
            if which_param == 'num_train':
                DataPath = DataPath_Parent + f"data_size={param}_deg={deg}_e={e}_d={d}_p={p}_x_dist={x_dist}_num_test={num_test}/"
            if which_param == 'deg':
                DataPath = DataPath_Parent + f"data_size={num_train}_deg={param}_e={e}_d={d}_p={p}_x_dist={x_dist}_num_test={num_test}/"
            if which_param == 'e':
                DataPath = DataPath_Parent + f"data_size={num_train}_deg={deg}_e={param}_d={d}_p={p}_x_dist={x_dist}_num_test={num_test}/"
            if which_param == 'd':
                DataPath = DataPath_Parent + f"data_size={num_train}_deg={deg}_e={e}_d={param}_p={p}_x_dist={x_dist}_num_test={num_test}/"
            if which_param == 'p':
                DataPath = DataPath_Parent + f"data_size={num_train}_deg={deg}_e={e}_d={d}_p={param}_x_dist={x_dist}_num_test={num_test}/"
            print(DataPath)
            cost_Oracle_Post_all,cost_Oracle_Ante_all,cost_OLS_Post_all,cost_OLS_Ante_all,cost_DDR_Post_all,cost_DDR_Ante_all = Data_LSM.load_cost_data(DataPath)
            h2h_[_index], regret_[_index] = self.calculate_h2h_regret(mu,lamb,iteration_all,\
                                cost_DDR_Post_all,cost_OLS_Post_all,cost_Oracle_Post_all,\
                                    cost_DDR_Ante_all,cost_OLS_Ante_all,cost_Oracle_Ante_all)
        return regret_, h2h_