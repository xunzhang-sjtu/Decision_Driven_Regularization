import numpy as np
import time
from gurobipy import *
import Utils

def sub_jero(data, c_train,thres, r, rho, mu, tau):
    x_train = data[3]
    z_train = data[5]
    N,p = x_train.shape
    N,d = z_train.shape
    
    m = Model('subprob_jero')
    m.setParam('OutputFlag', 0)

    z = np.minimum(c_train, thres)

    #add variables
    lambd_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])
    y_ind = tuplelist([(n,i)  for n in range(N) for i in range(d)])
    alpha_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])
    theta_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])

    lambd = m.addVars(lambd_ind, lb= -GRB.INFINITY, ub = GRB.INFINITY)
    y = m.addVars(y_ind, lb=0)
    gamma = m.addVar(vtype = GRB.CONTINUOUS,lb= 0, ub = GRB.INFINITY)
    t = m.addVar(vtype = GRB.CONTINUOUS,lb= -GRB.INFINITY, ub = GRB.INFINITY)
    alp =  m.addVars(alpha_ind, lb=0)
    theta = m.addVars(theta_ind, lb =0)

    m.setObjective(t - tau, GRB.MINIMIZE)

    #add constraints
    for n in range(N):
        m.addConstr(quicksum(y[n,i] for i in range(d)) ==1)

    obj = 0
    lambd2 = 0
    theta2 = 0

    for i in range(d):
        m.addConstr( quicksum(lambd[n,i] for n in range(N)) == quicksum(alp[n,i] for n in range(N)) )
        for j in range(p):
        #dual constraint over the coefficient beta
            m.addConstr( quicksum(x_train[n,j]*alp[n,i] for n in range(N)) == \
                        quicksum(x_train[n,j]*lambd[n,i]  for n in range(N)) )
        #dual constraint over the intercept
        for n in range(N):
            obj += z_train[n,i]*lambd[n,i]
            theta2 += thres*theta[n,i]
            lambd2 += lambd[n,i]*lambd[n,i]
            m.addConstr( (1-mu)*y[n,i] == alp[n,i]+theta[n,i] )

    m.addConstr( mu*quicksum(z[n,j]*y[n,i] for n in range(N) for i in range(d)) + gamma + obj + theta2 <= N*t )                 
    m.addQConstr( lambd2*(rho + r*N) <= gamma*gamma )
    m.addConstr( gamma >= 0 )

    m.write('model_subjero.rlp')
    m.optimize()


    y_temp = m.getAttr('x', y)
    lamb_temp = m.getAttr('x', lambd)
    theta_temp = m.getAttr('x', theta)
    alp_temp = m.getAttr('x', alp)

    y_results = []
    t_results = t.X
    lamb_results = []
    theta_results = []
    alpha_results = []
    for n in range(N):
        y_results.append([y_temp[(n,i)] for i in range(d)])
        lamb_results.append([lamb_temp[(n,i)] for i in range(d)])
        theta_results.append([theta_temp[(n,i)] for i in range(d)])
        alpha_results.append([alp_temp[(n,i)] for i in range(d)])

    gamma_results = gamma.X

    return m.objVal, y_results, m.objVal+tau, lamb_results, theta_results, alpha_results, gamma_results

def bisection(data, thres, mu, tau, rho,r_1, r_2, r_tol):
    x_train = data[3]
    z_train = data[5]
    N,d = z_train.shape

    while r_2 - r_1 >= r_tol:
        r =(r_1 + r_2)/2
        value, y , t_results, lamb, theta, alp, gamma= sub_jero(data, thres, r, rho, mu, tau)
        if value <= 0:
            r_1 = r
        else:
            r_2 = r

    r = r_1

    value, y , t_results, lamb, theta, alp, gamma= sub_jero(data, thres, r, rho, mu, tau)
    obj2 = 0
    for n in range(N):
        for i in range(d):
            obj2 += lamb[n][i]*lamb[n][i]
    r_jero = r
    y_jero = y
    t_jero = t_results
    t_minus_tau = t_results - tau
    final_lamb = 2/np.sqrt(obj2/(rho + N*r_jero))

    print(r_jero)

    return y_jero, r_jero,t_jero,t_minus_tau, lamb, theta, alp, gamma, final_lamb

## JERO will not change with lambda because here we fix the target tau
def jero(data,x_train,samples_train,d,p,thres,rho,r,y_jero):
    # DDR get the worst beta

    m = Model("JERO")
    m.setParam('OutputFlag', 0)
    N = samples_train

    W_ind = tuplelist([(i,j) for i in range(d) for j in range(p)])
    w0_ind = tuplelist([i for i in range(d)])
    auxi_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])

    W_jero = m.addVars(W_ind, lb=-GRB.INFINITY)
    w0_jero = m.addVars(w0_ind, lb=-GRB.INFINITY)
    auxi_jero = m.addVars(auxi_ind, lb = -GRB.INFINITY, ub = thres)

    m.addConstrs((quicksum([x_train[n,j]*W_jero[i,j] for j in range(p)]) + w0_jero[i] >= auxi_jero[n,i] \
                 for n in range(N) for i in range(d) ))
    m.addConstr(Utils.Loss(data, d, p, W_jero, w0_jero) <= rho/len(x_train) + r)

    temp = 0
    for n in range(N): 
        for i in range(d):
            temp += y_jero[n][i] * auxi_jero[n,i]
    temp = temp / N
    m.setObjective(temp, GRB.MAXIMIZE)

    m.optimize()


    # W = m.getAttr('x', beta_jero)
    # w0 = m.getAttr('x', inter_jero)
    # **** above is the original code, replacing by the following codes *******
    W = m.getAttr('x', W_jero)
    w0 = m.getAttr('x', w0_jero)


    W_results = []
    for i in range(d):
        W_results.append([W[(i,j)] for j in range(p)])
    w0_results = [w0[j] for j in range(d)]

    return W_results, w0_results


def jeroreal_solver(x_train, z_train, r, Lols, thres, x):
#     x_train = data[3]
    #### If we threshold here, we should threshold for OLS and other models also.
#     z_train = data[5] 
    z_train_min = np.minimum(z_train,thres)
    N,p = x_train.shape
    N,d = z_train.shape

    m = Model('jero_real')
    m.setParam('OutputFlag', 0)


    #add variables
    lambd_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])
    y_ind = tuplelist([i for i in range(d)])
    alph_ind = tuplelist([i for i in range(d)])
    beta_ind = tuplelist([i for i in range(d)])


    lambd = m.addVars(lambd_ind, lb= -GRB.INFINITY, ub = GRB.INFINITY)
#     y = m.addVars(y_ind, vtype = GRB.BINARY)
    y = m.addVars(y_ind, lb = 0, ub = 1)
    gamma = m.addVar(vtype = GRB.CONTINUOUS,lb = 0, ub = GRB.INFINITY)
    alph =  m.addVars(beta_ind, lb = 0)
    beta =  m.addVars(beta_ind, lb = 0)
    
    # Objective
    m.setObjective( gamma*np.sqrt(Lols + N*r) + quicksum(beta[i]*thres for i in range(d)) + quicksum(lambd[n,i]*z_train[n,i] for n in range(N) for i in range(d)), GRB.MINIMIZE )

    # Constraints
    m.addConstr( quicksum( y[i] for i in range(d)) == 1 )
    m.addConstrs( y[i] == alph[i] + beta[i] for i in range(d) )
    

    lambd2 = 0
    for i in range(d):
        m.addConstr( alph[i] == quicksum(lambd[n,i] for n in range(N)) )
        for j in range(p):
            m.addConstr( x[j]*alph[i] == quicksum(lambd[n,i]*x_train[n,j] for n in range(N)) )
        for n in range(N):
            lambd2 += lambd[n,i]*lambd[n,i]

    m.addQConstr( lambd2 <= gamma*gamma )
#     m.addConstr( gamma >= 0 )

#     m.write('model_jero_real.rlp')
    
    # Solve the model        
    m.optimize()
    
    # Obtain results
    y_temp = m.getAttr('x', y)
    y_results = []
    for i in range(d):
        y_results.append(y_temp[i])

    return m.objVal, y_results

def obtain_y_jero_real(x_train, z_train, x_test, r, Lols, thres):
#     x_test = data[0]
    y = []
    for m in range(len(x_test)):
        obj_temp, y_temp = jeroreal_solver(x_train, z_train, r, Lols, thres, x_test[m])
        y.append(y_temp)
    return y

def L1_jeroreal_solver(x_train, z_train, r, Lols, thres, x):
#     x_train = data[3]
    #### If we threshold here, we should threshold for OLS and other models also.
#     z_train = data[5] 
    N,p = x_train.shape
    N,d = z_train.shape

    m = Model('jero_real')
    m.setParam('OutputFlag', 0)


    #add variables
    Lambd_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])
    Phi_ind = tuplelist([(n,i) for n in range(N) for i in range(d)])
    y_ind = tuplelist([i for i in range(d)])
    alph_ind = tuplelist([i for i in range(d)])
    beta_ind = tuplelist([i for i in range(d)])


    Lambd = m.addVars(Lambd_ind, lb = 0, ub = GRB.INFINITY)
    Phi = m.addVars(Lambd_ind, lb = 0, ub = GRB.INFINITY)
    y = m.addVars(y_ind, vtype = GRB.BINARY)
#     y = m.addVars(y_ind, lb = 0, ub = 1)
    gamma = m.addVar(vtype = GRB.CONTINUOUS,lb = 0, ub = GRB.INFINITY)
    alph =  m.addVars(alph_ind, lb = 0)
    beta =  m.addVars(beta_ind, lb = 0)
    
    # Objective
    m.setObjective( gamma*(Lols + N*r) + quicksum(beta[i]*thres for i in range(d)) +\
                   quicksum((Phi[n,i] - Lambd[n,i])*z_train[n,i] for n in range(N) for i in range(d)), GRB.MINIMIZE )

    # Constraints
    m.addConstr(quicksum( y[i] for i in range(d)) == 1 )
    
    m.addConstrs( y[i] == alph[i] + beta[i] for i in range(d) )

    for i in range(d):
        m.addConstr( alph[i] == quicksum(Phi[n,i] - Lambd[n,i] for n in range(N)) )
        for j in range(p):
            m.addConstr( x[j]*alph[i] == quicksum((Phi[n,i] - Lambd[n,i])*x_train[n,j] for n in range(N)) )

    m.addConstrs( Phi[n,i] + Lambd[n,i] == gamma for n in range(N) for i in range(d) )
    
    # Solve the model
    m.optimize()
    
    # Obtain results
    y_temp = m.getAttr('x', y)
    y_results = []
    for i in range(d):
        y_results.append(y_temp[i])

    return m.objVal, y_results

def L1_obtain_y_jero_real(x_train, z_train, x_test, r, Lols, thres):
#     x_test = data[0]
    y = []
    for m in range(len(x_test)):
        obj_temp, y_temp = L1_jeroreal_solver(x_train, z_train, r, Lols, thres, x_test[m])
        y.append(y_temp)
    return np.rint(y)