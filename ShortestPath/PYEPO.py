# build optModel
from pyepo.model.grb import optGrbModel
import numpy as np
import pickle
import pyepo
from torch import nn
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class shortestPathModel(optGrbModel):

    def __init__(self,grid):
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__()

    def _getArcs(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        import gurobipy as gp
        from gurobipy import GRB
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self.arcs, name="x")
        # sense
        m.modelSense = GRB.MINIMIZE
        # flow conservation constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        return m, x
    


# build linear model
class LinearRegression(nn.Module):

    def __init__(self,num_feat,grid):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, (grid[0]-1)*grid[1]+(grid[1]-1)*grid[0])

    def forward(self, x):
        out = self.linear(x)
        return out

class PyEPO_Method:

    def __init__(self):
        pass
    # train model
    def Implement_PyEPO(self,arcs,grid,num_feat, loss_func, method_name, loader_train,num_epochs, lr):
        import time
        import torch
        from Peformance import performance_evaluation
        perfs = performance_evaluation()
        
        # init model
        reg = LinearRegression(num_feat,grid)
        # set adam optimizer
        optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
        # train mode
        reg.train()

        # init elpased time
        elapsed = 0 
        epoch = 0
        prev_cost_mean = 1e8  # 记录前一次的cost均值
        while True:
            # start timing
            tick = time.time()
            # load data
            for i, data in enumerate(loader_train):
                x, c, w, z = data
                # cuda
                if torch.cuda.is_available():
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                cp = reg(x)
                if method_name == "spo+":
                    loss = loss_func(cp, c, w, z)
                if method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                    loss = loss_func(cp, w)
                if method_name in ["dbb", "nid"]:
                    loss = loss_func(cp, c, z)
                if method_name in ["pg", "ltr"]:
                    loss = loss_func(cp, c)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record time
                tock = time.time()
                elapsed += tock - tick
            
            W_EPO_mat = (reg.linear.weight.data).numpy()
            w0_EPO = (reg.linear.bias.data).numpy()

            cost_arr = perfs.compute_EPO_Cost(reg, loader_train,arcs,grid)
            # 终止条件计算
            diff = abs(np.nanmean(cost_arr) - prev_cost_mean)
            # print("epoch = ",epoch,"in sample cost = ",np.mean(cost_arr),",diff = ",diff)
            if epoch > num_epochs or diff < 0.00001:
                break  # 终止循环
            else:
                prev_cost_mean = np.nanmean(cost_arr)
            epoch = epoch + 1
        return cost_arr,W_EPO_mat,w0_EPO


    def run(self,method_names,DataPath_seed,batch_size,num_feat,grid,num_epochs,x_train,c_train,arcs):
        # 在这里实例化 shortestpathModel
        optmodel = shortestPathModel(grid)
        # arcs = optmodel.arcs

        # get optDataset
        dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
        # set data loader
        from torch.utils.data import DataLoader
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

        rst_EPO = {}
        for method_name in method_names:
            if method_name == "spo+":
                # print("=========== Run SPO,",DataPath_seed)
            # Obtain SPO cost
                spop = pyepo.func.SPOPlus(optmodel, processes=2)
                cost_SPO,W_EPO_mat,w0_EPO = self.Implement_PyEPO(arcs,grid,num_feat, spop, method_name, loader_train,num_epochs, 1e-3)
                rst_EPO["SPO"] = cost_SPO

            if method_name == "pg":
                # print("=========== Run PG,",DataPath_seed)
                pg = pyepo.func.perturbationGradient(optmodel, sigma=0.1, two_sides=False, processes=2)
                cost_PG,W_EPO_mat,w0_EPO = self.Implement_PyEPO(arcs,grid,num_feat, pg, method_name, loader_train,num_epochs, 1e-3)
                rst_EPO["PG"] = cost_PG

            if method_name == "ltr":
                # print("=========== Run LTR,",DataPath_seed)
                ptltr,W_EPO_mat,w0_EPO = pyepo.func.pointwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset_train)
                cost_LTR = self.Implement_PyEPO(arcs,grid,num_feat, ptltr, method_name, loader_train,num_epochs, 1e-3)
                rst_EPO["LTR"] = cost_LTR

        return W_EPO_mat,w0_EPO