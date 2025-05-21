import numpy as np
from gurobipy import *

class DDR_Problem_Evaluation:
    def __init__(self):
        pass
            
    def compute_DDR_out_of_sample_Cost(self,c_pred,c_oracle_avg,noise,data_generation_process,with_noise):
        if data_generation_process == "DDR_Data_Generation":
            cost_pred_arr = []
            if with_noise:
                for j in range(np.shape(c_pred)[0]):
                    opt_index = np.argmin(c_pred[j,:])
                    c_oralce_realization = c_oracle_avg[j,opt_index] + noise[:,opt_index]
                    cost_real = np.nanmean(c_oralce_realization)
                    cost_pred_arr.append(cost_real)
            else:
                for j in range(np.shape(c_pred)[0]):
                    opt_index = np.argmin(c_pred[j,:])
                    cost_real = c_oracle_avg[j,opt_index]
                    cost_pred_arr.append(cost_real)

        return np.asarray(cost_pred_arr)
    
    def decision_finder(self,z):
        N,d = z.shape
        y = np.float64((z == np.min(z, axis = np.array(z).ndim - 1, keepdims=True)).view('i1'))
        mask = np.all(y == np.array([1 for i in range(d)]), axis = np.array(z).ndim - 1) 
        y[mask] = [1/d for i in range(d)]  ## if multiple minimum, assign equal prob.

        return np.array(y)
    
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
        cost_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base))
        regret_reduction = (np.mean(c_diff))/np.abs(np.mean(c_base) - np.mean(c_oracle))
        return lbel, win_ratio, cost_reduction, regret_reduction