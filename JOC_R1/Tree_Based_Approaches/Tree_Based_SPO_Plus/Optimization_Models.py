
from gurobipy import *
import numpy as np

class shortestPathModel:

    def __init__(self,dim,mu,lamb):
        Edge_list,Edge_dict = self.network_edges(dim)
        self.Edge_list = Edge_list
        self.Edge_dict = Edge_dict
        self.dim = dim

        m_shortest_path,flow,Edges = self._getModel()
        self.model = m_shortest_path
        self.flow = flow
        self.Edges = Edges

        self.mu = mu
        self.lamb = lamb

    def network_edges(self,dim):
        # dim = 3 #(creates dim * dim grid, where dim = number of vertices)
        Edge_list = [(i,i+1) for i in range(1, dim**2 + 1) if i % dim != 0]
        Edge_list += [(i, i + dim) for i in range(1, dim**2 + 1) if i <= dim**2 - dim]
        Edge_dict = {} #(assigns each edge to a unique integer from 0 to number-of-edges)
        for index, edge in enumerate(Edge_list):
            Edge_dict[edge] = index
        # D = len(Edge_list) # D = number of decisions
        return Edge_list,Edge_dict

    def _getModel(self):        
        Edge_list = self.Edge_list
        dim = self.dim

        Edges = tuplelist(Edge_list)
        # Find the optimal total cost for an observation in the context of shortes path
        m_shortest_path = Model('shortest_path')
        m_shortest_path.Params.OutputFlag = 0
        flow = m_shortest_path.addVars(Edges, ub = 1, name = 'flow')
        m_shortest_path.addConstrs((quicksum(flow[i,j] for i,j in Edges.select(i,'*')) - quicksum(flow[k, i] for k,i in Edges.select('*',i)) == 0 for i in range(2, dim**2)), name = 'inner_nodes')
        m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select(1, '*')) == 1), name = 'start_node')
        m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select('*', dim**2)) == 1), name = 'end_node')

        return m_shortest_path,flow,Edges
    

    def solve_shortest_path(self,cost):
        m_shortest_path = self.model
        flow = self.flow
        Edge_dict = self.Edge_dict
        Edges = self.Edges
        
        # m_shortest_path.setObjective(quicksum(flow[i,j] * cost[Edge_dict[(i,j)]] for i,j in Edges), GRB.MINIMIZE)
        m_shortest_path.setObjective(LinExpr( [ (cost[Edge_dict[(i,j)]],flow[i,j] ) for i,j in Edges]), GRB.MINIMIZE)
        m_shortest_path.optimize()
        return {'weights': m_shortest_path.getAttr('x', flow), 'objective': m_shortest_path.objVal}




    def solve_DDR_Model(self,C):
      N_obs,d = C.shape
      Edge_list = self.Edge_list
      dim = self.dim
      mu = self.mu 
      lamb = self.lamb

      m = Model("ddr")
      #m.setParam("DualReductions",0)
      m.setParam('OutputFlag', 0)
      num_routes = len(Edge_list) 
      num_nodes = dim*dim
      alpha = m.addVars(N_obs,num_nodes,lb=-GRB.INFINITY,name="alpha")
      c_hat = m.addVars(num_routes,lb=-GRB.INFINITY,name="c_hat")
      expr_obj = 0
      err = []
      for n in range(N_obs):
          cost_true_tem = C[n]
          expr_obj = expr_obj + alpha[n,num_nodes-1] - alpha[n,0]
          for ind in range(num_routes):
              err.append(cost_true_tem[ind] - c_hat[ind])
              e = Edge_list[ind]
              j = e[1]
              i = e[0]
              # print("j = ",j,", i = ",i, ", e = ",e)
              m.addConstr(alpha[n,j-1] - alpha[n,i-1] >= -mu*cost_true_tem[ind] - (1-mu)*c_hat[ind])

      m.setObjective(quicksum([err[k] * err[k] for k in range(len(err))]) + (lamb/N_obs)*(expr_obj), GRB.MINIMIZE)
      m.optimize()

      c_dict = m.getAttr("x",c_hat)
      c_arr = [c_dict[ind] for ind in range(len(c_dict))]
      ddr_rst = {
        "obj": m.ObjVal,
        "sol": c_arr
      }
      # c_val= m.getAttr("x",c_hat)
      # alpha_rst = m.getAttr('x', alpha)
      return ddr_rst
