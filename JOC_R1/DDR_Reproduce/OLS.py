import numpy as np
from gurobipy import *
import time
from rsome import ro
from rsome import grb_solver as grb
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
import warnings
import Utils


def ols_solver(x_train, z_train):
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    
    start = time.time()
    m = Model("OLS")
    #m.setParam("DualReductions",0)
    m.setParam('OutputFlag', 0)

    W_ind = tuplelist([(i,j) for i in range(d) for j in range(p)])
    w0_ind = tuplelist([i for i in range(d)])
    
    W_ddr = m.addVars(W_ind, lb=-GRB.INFINITY)
    
    w0_ddr = m.addVars(w0_ind, lb=-GRB.INFINITY)

    m.setObjective(Utils.Loss(x_train, z_train, W_ddr, w0_ddr), GRB.MINIMIZE)

    m.optimize()
    end = time.time()

    W = m.getAttr('x', W_ddr)
    w0 = m.getAttr('x', w0_ddr)
    W_results = []
    for i in range(d):
        W_results.append([W[(i,j)] for j in range(p)])
    w0_results = [w0[j] for j in range(d)]
    end = time.time()
    return W_results, w0_results, end-start, m.ObjVal

def L1_ols_solver(x_train, z_train):
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    
    start = time.time()
    m = Model("OLS")
    #m.setParam("DualReductions",0)
    m.setParam('OutputFlag', 0)

    W_ind = tuplelist( [(i,j) for i in range(d) for j in range(p)] )
    w0_ind = tuplelist( [i for i in range(d)] )
    obj_ind = tuplelist( [(n,i) for n in range(N) for i in range(d)] )
    
    W_ddr = m.addVars( W_ind, lb=-GRB.INFINITY )
    w0_ddr = m.addVars( w0_ind, lb=-GRB.INFINITY )
    obj_ddr = m.addVars( obj_ind, lb=0 )

    m.setObjective( quicksum(obj_ddr[n, i] for n in range(N) for i in range(d)), GRB.MINIMIZE )
    
    m.addConstrs( obj_ddr[n,i] >= z_train[n,i] - quicksum(x_train[n][j]*W_ddr[i,j] for j in range(p)) \
                 - w0_ddr[i] for n in range(N) for i in range(d) )
    m.addConstrs( obj_ddr[n,i] >= - z_train[n,i] + quicksum(x_train[n][j]*W_ddr[i,j] for j in range(p)) \
                 + w0_ddr[i] for n in range(N) for i in range(d) )

    m.optimize()
    end = time.time()

    W = m.getAttr('x', W_ddr)
    w0 = m.getAttr('x', w0_ddr)
    W_results = []
    for i in range(d):
        W_results.append( [W[(i,j)] for j in range(p)] )
    w0_results = [w0[j] for j in range(d)]
    end = time.time()
    return W_results, w0_results, end-start, m.ObjVal/N

def L1_ols_solver_rsome(x_train, z_train):
#     x_train = data[3]
#     z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    
    start = time.time()
    m = ro.Model('abs')
    
    W = m.dvar( (d, p) )
    w0 = m.dvar( d )
    obj = m.dvar( (N, d) )

    m.min( obj.sum() )
    
#     m.st( obj >= z_train - x_train @ W_ddr - np.ones((N,d))*w0 )
#     m.st( obj >= - z_train + x_train @ W_ddr + np.ones((N,d))*w0 )
    for n in range(N):
        for i in range(d):
            m.st( obj[n,i] >= z_train[n,i] - (x_train[n,:]*W[i,:]).sum() - w0[i] )
            m.st( obj[n,i] >= - z_train[n,i] + (x_train[n,:]*W[i,:]).sum() + w0[i] )

    m.solve(grb)
    end = time.time()

    W_results = W.get()
    w0_results = w0.get()
    obj = obj.get()
    end = time.time()
    return W_results, w0_results, end-start, obj.sum()/N


def lasso_solver(x_train, z_train):
#     x_train = data[3]
#     z_train = data[5]
    start = time.time()
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = MultiTaskLassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
    # fit model
    warnings.filterwarnings("ignore")
    model.fit(x_train, z_train)
    W_results = model.coef_
    w0_results = model.intercept_
    end = time.time()
    return W_results, w0_results, end-start

def ridge_solver(x_train, z_train):
#     x_train = data[3]
#     z_train = data[5]
    start = time.time()
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = RidgeCV(alphas=np.arange(0, 5, 0.1), fit_intercept=True, cv=cv)
    # fit model
    warnings.filterwarnings("ignore")
    model.fit(x_train, z_train)
    W_results = model.coef_
    w0_results = model.intercept_
    end = time.time()
    return W_results, w0_results, end-start