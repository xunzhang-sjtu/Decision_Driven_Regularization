import numpy as np
import time
from gurobipy import *

class performance_evaluation:
    def __init__(self):
        pass

    def compute_Cost_with_Prediction(self,arcs,w0,W, grid,c,feature):
        from Shortest_Path_Model import My_ShortestPathModel
        full_shortest_model = My_ShortestPathModel()
        # evaluate
        cost_pred_arr = []
        # load data
        # print("Feature Shape = ",np.shape(feature)[0])
        for j in range(np.shape(feature)[0]):
            cost = W @ feature[j,:] + w0
            sol_pred = full_shortest_model.solve_Shortest_Path(arcs,cost,grid)
            cost_pred = np.dot(sol_pred, c[j])
            cost_pred_arr.append(cost_pred)
        # print("Average OLS Cost = ", np.mean(cost_pred_arr))
        return cost_pred_arr
            
    def compute_DDR_out_of_sample_Cost(self,arcs, grid,c_pred,c_oracle_avg,noise):
        from Shortest_Path_Model import My_ShortestPathModel
        full_shortest_model = My_ShortestPathModel()
        # evaluate
        cost_pred_arr = []
        for j in range(np.shape(c_pred)[0]):
            sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
            c_oralce_realization = c_oracle_avg[j,:] + noise
            cost_real = np.nanmean(c_oralce_realization @ sol_pred)
            cost_pred_arr.append(cost_real)

        return cost_pred_arr
    
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

        return cost_pred_arr


    # from pyepo import EPO
    def compute_EPO_Cost(self,predmodel, dataloader,arcs,grid):
        from Shortest_Path_Model import My_ShortestPathModel
        import torch
        full_shortest_model = My_ShortestPathModel()
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



class DDR_Problem_Evaluation:
    def __init__(self):
        pass
            
    def compute_DDR_out_of_sample_Cost(self,c_pred,c_oracle_avg,noise):

        # evaluate
        cost_pred_arr = []
        for j in range(np.shape(c_pred)[0]):
            opt_index = np.argmin(c_pred[j,:])
            c_oralce_realization = c_oracle_avg[j,opt_index] + noise[:,opt_index]
            cost_real = np.nanmean(c_oralce_realization)
            cost_pred_arr.append(cost_real)

        return cost_pred_arr
    
    # def compute_SPO_out_of_sample_Cost(self,arcs, grid,c_pred,c_oracle_avg,noise):
    #     from Shortest_Path_Model import My_ShortestPathModel
    #     full_shortest_model = My_ShortestPathModel()
    #     # evaluate
    #     cost_pred_arr = []
    #     for j in range(np.shape(c_pred)[0]):
    #         sol_pred = full_shortest_model.solve_Shortest_Path(arcs,c_pred[j],grid)
    #         c_oralce_realization = c_oracle_avg[j,:] * noise
    #         cost_real = np.nanmean(c_oralce_realization @ sol_pred)
    #         cost_pred_arr.append(cost_real)

    #     return cost_pred_arr


    # # from pyepo import EPO
    # def compute_EPO_Cost(self,predmodel, dataloader,arcs,grid):
    #     from Shortest_Path_Model import My_ShortestPathModel
    #     import torch
    #     full_shortest_model = My_ShortestPathModel()
    #     # evaluate
    #     predmodel.eval()
    #     cost_pred_arr = []
    #     # load data
    #     for data in dataloader:
    #         x, c, w, z = data
    #         # cuda
    #         if next(predmodel.parameters()).is_cuda:
    #             x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
    #         # predict
    #         with torch.no_grad(): # no grad
    #             cp = predmodel(x).to("cpu").detach().numpy()
    #         # print("cp[0] = ",cp[0])
    #         # solve
    #         for j in range(cp.shape[0]):
    #             sol_pred = full_shortest_model.solve_Shortest_Path(arcs,cp[j],grid)
    #             cost_pred = np.dot(sol_pred, c[j].to("cpu").detach().numpy())
    #             cost_pred_arr.append(cost_pred)
    #             # cost_true_arr.append(z[j].item())
    #     # turn back train mode
    #     predmodel.train()
    #     # normalized
    #     return cost_pred_arr
