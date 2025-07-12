from Optimization_Models import shortestPathModel
import numpy as np


def compute_oof_cost(c_pred,c_test,dim,mu,lamb):
    SPM = shortestPathModel(dim,mu,lamb)
    Edge_dict = SPM.Edge_dict
    Edges = SPM.Edges
    [N_obs, N_routes] = np.shape(c_test)
    cost_ = []
    for n in range(N_obs):
        c_this = c_test[n,:]
        rst = SPM.solve_shortest_path(c_pred[n,:])
        sol = rst["weights"]
        cost_tem = sum([(c_this[Edge_dict[edge]]* sol[edge] ) for edge in Edges])
        cost_.append(cost_tem)
    return np.asarray(cost_)
