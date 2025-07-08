import numpy as np 
import decision_problem_solver as dps

def get_Edges(dim):
    # dim = 3 #(creates dim * dim grid, where dim = number of vertices)
    Edge_list = [(i,i+1) for i in range(1, dim**2 + 1) if i % dim != 0]
    Edge_list += [(i, i + dim) for i in range(1, dim**2 + 1) if i <= dim**2 - dim]
    Edge_dict = {} #(assigns each edge to a unique integer from 0 to number-of-edges)
    for index, edge in enumerate(Edge_list):
        Edge_dict[edge] = index
    return Edge_list,Edge_dict


def compute_out_of_sample_cost(deg,W_star,x_test,c_test,num_feat,dim):
    cost_oracle_ori = (W_star @ x_test.T)/np.sqrt(num_feat) + 3
    cost_oracle_pred = (cost_oracle_ori ** deg + 1).T
    # print("Oracle Pred = ", np.round(cost_oracle_pred,4))
    Edge_list,Edge_dict = get_Edges(dim)
    [N_obs, N_routes] = np.shape(cost_oracle_pred)
    cost_Oracle = []
    for n in range(N_obs):
        decision_dict = dps.shortest_path(cost_oracle_pred[n,:])['weights']
        decision_arr = np.asarray([decision_dict[d_key] for d_key in Edge_list])
        cost_Oracle.append(decision_arr @ cost_oracle_pred[n,:])

    return np.asarray(cost_Oracle),cost_oracle_pred


def compute_out_of_sample_cost_tree(c_test,dim):

    Edge_list,Edge_dict = get_Edges(dim)
    [N_obs, N_routes] = np.shape(c_test)
    cost_Oracle = []
    for n in range(N_obs):
        decision_dict = dps.shortest_path(c_test[n,:])['weights']
        decision_arr = np.asarray([decision_dict[d_key] for d_key in Edge_list])
        cost_Oracle.append(decision_arr @ c_test[n,:])

    return np.asarray(cost_Oracle)