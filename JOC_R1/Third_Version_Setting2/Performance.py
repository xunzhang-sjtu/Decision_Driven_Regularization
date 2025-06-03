import numpy as np
from gurobipy import *
import time

def param_prediction_and_cost_estimation(x_test, W, w0, thres, yconstraint = 0, A = 0, b = 0, yB = 0, B = 0):
    M = len(x_test)
#     z_predict = np.minimum(x_test @ np.array(W).T + np.tile(w0, (M,1)),thres)
    z_predict = x_test @ np.array(W).T + np.tile(w0, (M,1))
    y_predict = decision_finder(z_predict, yconstraint, A, b, yB, B)
    c_predict = np.sum( np.minimum(z_predict, thres) * y_predict, axis = 1 )
    return np.array(z_predict), np.array(y_predict), np.array(c_predict)

def decision_finder(z, yconstraint = 0, A = 0, b = 0, yB = 0, B = 0):
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

def cross_compare2(c_item, c_base, c_oracle):
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

def cross_compare2plus(c_item, c_base, c_oracle):
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

def cross_compare(y_test_ddr, y_test_spo, z_test_ori, y_test_opt, d,thres):
    difference = 0
    powers = [2**i for i in range(d)]
    c_spo_true = np.sum(np.minimum(z_test_ori,thres) * y_test_spo, axis = 1)
    c_ddr_true = np.sum(np.minimum(z_test_ori,thres) * y_test_ddr, axis = 1)
    c_oracle = np.sum(np.minimum(z_test_ori,thres) * y_test_opt, axis = 1)
    std_spo_true = np.std(c_spo_true)
    std_ddr_true = np.std(c_ddr_true)
    ddr_wins = 0
    spo_wins = 0
    for i in range(len(y_test_ddr)):
        diff_vec = [y_test_spo[i][j] - y_test_ddr[i][j] for j in range(d)]
        if np.dot(powers, diff_vec) != 0:
            difference += 1
            if c_ddr_true[i] < c_spo_true[i]:
                ddr_wins += 1
            elif c_spo_true[i] < c_ddr_true[i]:
                spo_wins += 1
    if difference == 0:
        win_ratio = 0
    else:
        win_ratio = ddr_wins / difference
    return difference / len(y_test_ddr), win_ratio, (np.mean(c_ddr_true) - np.mean(c_spo_true))/np.abs(np.mean(c_oracle)),\
            (np.mean(c_ddr_true)-np.mean(c_oracle))/np.abs(np.mean(c_oracle)),(np.mean(c_ddr_true) - np.mean(c_spo_true))/std_ddr_true

def head_2_head(item, base):
    count = 0
    diff = 0
    N = len(item)
    for i in range(N):
        if item[i] < base[i]:
            count += 1
            diff += 1
        elif item[i] > base[i]:
            diff += 1
    if diff == 0:
        return 0,0
    else:
        return count / diff, diff / N
    
# def cross_compare1(y_item, y_base, c_item, c_base, c_oracle):
#     N = len(y_item)
#     y_diff = y_item - y_base
#     c_diff = c_item - c_base
#     lbel = np.zeros((N,1))
#     wins = 0
#     equals = 0
#     lose = 0
#     for n in range(N):
#         if np.sum(np.absolute(y_diff[n])) < 0.1: ## if same
#             equals += 1
#             lbel[n] = 0
#         else:
#             if c_diff[n] < 0:
#                 wins += 1
#                 lbel[n] = 1
#             else:
#                 lose += 1
#                 lbel[n] = -1
#     print(N, equals, wins, lose)
#     if N == equals:
#         win_ratio = 0
#     else:
#         win_ratio = wins/(N - equals)
#     mci = (np.mean(c_item) - np.mean(c_base))/np.abs(np.mean(c_oracle))
#     return lbel, win_ratio, mci