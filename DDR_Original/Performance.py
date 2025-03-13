import numpy as np
import time
from gurobipy import *

class performance_evaluation:
    def __init__(self):
        pass


    def decision_finder(self,z, yconstraint = 0, A = 0, b = 0, yB = 0, B = 0):
        N,d = z.shape
        if yconstraint == 0:
            if yB == 0:
                y = np.float64((z == np.min(z, axis = np.array(z).ndim - 1, keepdims=True)).view('i1'))
                mask = np.all(y == np.array([1 for i in range(d)]), axis = np.array(z).ndim - 1) 
                y[mask] = [1/d for i in range(d)]  ## if multiple minimum, assign equal prob.
            else:
                z_new = (np.array(B) @ np.array(z.T)).T
                y = np.float64((z_new == np.min(z_new, axis = np.array(z_new).ndim - 1, keepdims=True)).view('i1'))
                mask = np.all(y == np.array([1 for i in range(d)]), axis = np.array(z_new).ndim - 1) 
                y[mask] = [1/d for i in range(d)]  ## if multiple minimum, assign equal prob.
        else:
            a,d= A.shape
            m = Model("TSP")
            #m.setParam("DualReductions",0) 
            m.setParam('OutputFlag', 0)

            y_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])

            y_tsp = m.addVars(y_ind, vtype=GRB.BINARY)

            m.setObjective( quicksum(z[n][i]*y_tsp[n,i] for i in range(d) for n in range(N)), GRB.MINIMIZE)
            m.addConstrs( quicksum( y_tsp[n,i] for i in range(d) ) == 1 for n in range(N) )
            m.addConstrs( quicksum( A[k,i]*y_tsp[n,i] for i in range(d) ) <= b[k] for k in range(a) for n in range(N) )
            m.optimize()
            end = time.time()

            y_tsp = m.getAttr('x', y_tsp)
            
            y = []
            for n in range(N):
                y.append([y_tsp[(n,i)] for i in range(d)])
        return np.array(y)




    def param_prediction_and_cost_estimation(self,x_test, W, w0, thres, yconstraint = 0, A = 0, b = 0, yB = 0, B = 0):
        M = len(x_test)
    #     z_predict = np.minimum(x_test @ np.array(W).T + np.tile(w0, (M,1)),thres)
        z_predict = x_test @ np.array(W).T + np.tile(w0, (M,1))
        y_predict = self.decision_finder(z_predict, yconstraint, A, b, yB, B)
        c_predict = np.sum( np.minimum(z_predict, thres) * y_predict, axis = 1 )
        return np.array(z_predict), np.array(y_predict), np.array(c_predict)
    
    def cross_compare2(self,c_item, c_base, c_oracle):
        N = len(c_item)
        c_diff = c_item - c_base
        lbel = np.zeros((N,1))
        
        equals = np.sum(c_diff == 0)
        wins = np.sum(c_diff < 0)
        lose = np.sum(c_diff > 0)
        
        lbel[c_diff < 0] = 1
        lbel[c_diff > 0] = -1
        
    #     print(N, equals, wins, lose)
        if N == equals:
            win_ratio = 0.5
        else:
            win_ratio = wins/(N - equals)
        mci = (np.mean(c_item) - np.mean(c_base))/np.abs(np.mean(c_oracle))
        return lbel, win_ratio, mci
    
    def cross_compare2plus(self,c_item, c_base, c_oracle):
        N = len(c_item)
        c_diff = c_item - c_base
        lbel = np.zeros((N,1))
        
        equals = np.sum(c_diff == 0)
        wins = np.sum(c_diff < 0)
        lose = np.sum(c_diff > 0)
        
        lbel[c_diff < 0] = 1
        lbel[c_diff > 0] = -1
        
    #     print(N, equals, wins, lose)
        if N == equals:
            win_ratio = 0.5
        else:
            win_ratio = wins/(N - equals)
        mci = (np.mean(c_item) - np.mean(c_base))/np.abs(np.mean(c_oracle))
    #     pio = (np.mean(c_item) - np.mean(c_oracle))/np.abs(np.mean(c_base) - np.mean(c_oracle))
        pio = (np.mean(c_base) - np.mean(c_item))/np.abs(np.mean(c_base) - np.mean(c_oracle))
        return lbel, win_ratio, mci, pio
        