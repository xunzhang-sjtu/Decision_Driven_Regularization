from gurobipy import *
from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
from rsome import cpt_solver as cpt
from Performance import performance_evaluation
perfs = performance_evaluation()
import numpy as np

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
    

class DDR_Processing:
    def __init__(self):
        self.ddr_object = DDR_method()

    def Implement_DDR(self,mu_all,lamb_all,arcs, grid,mis,bump,W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process):
        ddr_object = self.ddr_object
        num_nodes = grid[0] * grid[0]

        w0_ddr_dict = {}; W_ddr_dict = {}
        cost_DDR_Post = {}; cost_DDR_Ante = {}; RMSE_in_all = {};RMSE_out_all = {}
        for iter in iteration_all:
            for mu in mu_all:
                for lamb in lamb_all:
                    w0_ddr_dict[iter,mu,lamb],W_ddr_dict[iter,mu,lamb],alpha_rst,obj_ddr = ddr_object.solve_DDR(arcs,lamb,mu,num_nodes,x_train_all[iter],c_train_all[iter])
                    cost_pred = (W_ddr_dict[iter,mu,lamb] @ x_test_all[iter].T).T + w0_ddr_dict[iter,mu,lamb]
                    cost_in = (W_ddr_dict[iter,mu,lamb] @ x_train_all[iter].T).T + w0_ddr_dict[iter,mu,lamb]
                    RMSE_in_all[iter,mu,lamb] = np.sqrt(np.sum((cost_in - c_train_all[iter])**2)/len(cost_in[:,0]))
                    RMSE_out_all[iter,mu,lamb] = np.sqrt(np.sum((cost_pred - c_test_all[iter])**2)/len(cost_pred[:,0]))

                    if data_generation_process == "SPO_Data_Generation":
                        cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                        cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                        # cost_DDR_Post[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Post(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter])
                        cost_DDR_Ante[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_pred,cost_oracle_pred)

                    if data_generation_process == "DDR_Data_Generation":
                        cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                        cost_oracle_pred = (cost_oracle_ori ** mis).T
                        cost_DDR_Ante[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_pred,cost_oracle_pred)
                    # print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
            if iter % 20 == 0 and iter > 0:
                # print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Post =",np.nanmean(cost_DDR_Post[iter,mu,lamb]),
                #       ",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
                print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
        return cost_DDR_Post,cost_DDR_Ante,RMSE_in_all,RMSE_out_all
    
    def Implement_DDR_quad(self,mu_all,lamb_all,arcs, grid,mis,bump,W_star_all,x_test_all,c_test_all,x_train_all,c_train_all,iteration_all,num_feat,data_generation_process,x_train_quad_all,x_test_quad_all):
        ddr_object = self.ddr_object
        num_nodes = grid[0] * grid[0]

        w0_ddr_dict = {}; W_ddr_dict = {}
        cost_DDR_Post = {}; cost_DDR_Ante = {}; RMSE_in_all = {};RMSE_out_all = {}
        for iter in iteration_all:
            for mu in mu_all:
                for lamb in lamb_all:
                    w0_ddr_dict[iter,mu,lamb],W_ddr_dict[iter,mu,lamb],alpha_rst,obj_ddr = ddr_object.solve_DDR(arcs,lamb,mu,num_nodes,x_train_quad_all[iter],c_train_all[iter])
                    cost_pred = (W_ddr_dict[iter,mu,lamb] @ x_test_quad_all[iter].T).T + w0_ddr_dict[iter,mu,lamb]
                    cost_in = (W_ddr_dict[iter,mu,lamb] @ x_train_quad_all[iter].T).T + w0_ddr_dict[iter,mu,lamb]
                    RMSE_in_all[iter,mu,lamb] = np.sqrt(np.sum((cost_in - c_train_all[iter])**2)/len(cost_in[:,0]))
                    RMSE_out_all[iter,mu,lamb] = np.sqrt(np.sum((cost_pred - c_test_all[iter])**2)/len(cost_pred[:,0]))

                    if data_generation_process == "SPO_Data_Generation":
                        cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T)/np.sqrt(num_feat) + 3
                        cost_oracle_pred = (cost_oracle_ori ** mis + 1).T
                        # cost_DDR_Post[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Post(arcs, grid,cost_pred,cost_oracle_pred,noise_test_all[iter])
                        cost_DDR_Ante[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_pred,cost_oracle_pred)

                    if data_generation_process == "DDR_Data_Generation":
                        cost_oracle_ori = (W_star_all[iter] @ x_test_all[iter].T) + bump
                        cost_oracle_pred = (cost_oracle_ori ** mis).T
                        cost_DDR_Ante[iter,mu,lamb] = perfs.compute_SPO_out_of_sample_Cost_Ex_Ante(arcs, grid,cost_pred,cost_oracle_pred)
                    # print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
            if iter % 20 == 0 and iter > 0:
                # print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Post =",np.nanmean(cost_DDR_Post[iter,mu,lamb]),
                #       ",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
                print("DDR: iter=",iter,",mu=",mu,",lamb=",lamb,",cost_DDR_Ante =",np.nanmean(cost_DDR_Ante[iter,mu,lamb]))
        return cost_DDR_Post,cost_DDR_Ante,RMSE_in_all,RMSE_out_all