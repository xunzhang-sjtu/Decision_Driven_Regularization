from gurobipy import *
from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
from rsome import cpt_solver as cpt

class DDR_method():
    def __init__(self):
        pass

    def solve_DDR(self,arcs,lamb,mu_fixed,num_nodes,x_train,c_train):
        
        N,p = x_train.shape
        N,d = c_train.shape

        # DDR
        m = Model("ddr")
        #m.setParam("DualReductions",0)
        m.setParam('OutputFlag', 0)

        W_ind = tuplelist( [(i,j) for i in range(d) for j in range(p)] )
        w0_ind = tuplelist( [i for i in range(d)])

        W_ddr = m.addVars(W_ind, lb=-GRB.INFINITY,name = "W" )
        w0_ddr = m.addVars(w0_ind, lb=-GRB.INFINITY,name = "W0" )
        alpha = m.addVars(N,num_nodes,lb=-GRB.INFINITY,name="alpha")
        expr_obj = 0
        err = []
        for n in range(N):
            cost_true_tem = c_train[n]
            expr_obj = expr_obj + alpha[n,num_nodes-1] - alpha[n,0]
            for ind in range(len(arcs)):
                cost_pred_tem = quicksum([W_ddr[ind,j] * x_train[n,j] for j in range(p)]) + w0_ddr[ind]
                err.append(cost_true_tem[ind] - cost_pred_tem)
                e = arcs[ind]
                j = e[1]
                i = e[0]
                # print("j = ",j,", i = ",i, ", e = ",e)
                m.addConstr(alpha[n,j] - alpha[n,i] >= -mu_fixed*cost_true_tem[ind] - (1-mu_fixed)*cost_pred_tem)

        m.setObjective(quicksum([err[k] * err[k] for k in range(len(err))])/N + lamb*(expr_obj)/N, GRB.MINIMIZE)
        m.optimize()
        
        W_DDR_rst = m.getAttr('x', W_ddr)
        w0_DDR_rst = m.getAttr('x', w0_ddr)
        W_ddr_val = []
        for i in range(d):
            W_ddr_val.append([W_DDR_rst[(i,j)] for j in range(p)])
        w0_ddr_val = [w0_DDR_rst[i] for i in range(d)]

        alpha_rst = m.getAttr('x', alpha)
        return w0_ddr_val,W_ddr_val,alpha_rst,m.ObjVal