import time
from gurobipy import *

def obj1(x, W, y, d,w0):
    a = []
    N,p = x.shape
    for n in range(N):
        for i in range(d):
            temp = []
            for j in range(p):
                temp.append(x[n][j]*W[i,j])
            a.append((sum(temp) + w0[i])*y[n][i])
    return sum(a)/N

def spo_solver(x_train, z_train, opt_y):
    start = time.time()
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    
    mu = -1
    lamb = 1
    m = Model("spo+")
    m.setParam('OutputFlag', 0)

    W_ind = tuplelist([(i,j) for i in range(d) for j in range(p)])
    w0_ind = tuplelist([i for i in range(d)])
    t_ind = tuplelist([n for n in range(N)])

    W_spo = m.addVars(W_ind, lb=-GRB.INFINITY)
    w0_spo = m.addVars(w0_ind, lb=-GRB.INFINITY)
    t_spo = m.addVars(t_ind, lb=-GRB.INFINITY)

    m.setObjective(( lamb*((1-mu)*obj1(x_train, W_spo, opt_y, d,w0_spo) + quicksum([t_spo[n] for n in range(N)]) / N)), GRB.MINIMIZE )

    m.addConstrs(( -mu*z_train[n][i] - (1- mu)*(quicksum([x_train[n][j]*W_spo[i,j] for j in range(p) ]) + w0_spo[i]) -\
                  t_spo[n] <= 0 for n in range(N) for i in range(d) ))

    m.optimize()

    W = m.getAttr('x', W_spo)
    w0 = m.getAttr('x', w0_spo)
    W_results = []
    for i in range(d):
        W_results.append([W[(i,j)] for j in range(p)])
    w0_results = [w0[i] for i in range(d)]
    end = time.time()
    return W_results, w0_results, end-start