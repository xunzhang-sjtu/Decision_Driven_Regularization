# build optModel
from pyepo.model.grb import optGrbModel
import pyepo
import numpy as np
from Shortest_Path_Model import My_ShortestPathModel
import pickle

class shortestPathModel(optGrbModel):

    def __init__(self):
        self.grid = (5,5)
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
    

from torch import nn
# build linear model
class LinearRegression(nn.Module):

    def __init__(self,num_feat,grid):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, (grid[0]-1)*grid[1]+(grid[1]-1)*grid[0])

    def forward(self, x):
        out = self.linear(x)
        return out


class run_SPO_Shortest_Path:
    def __init__(self):
        pass

    # from pyepo import EPO
    def evaluation_SPO(self,predmodel, dataloader,arcs,grid):
        import torch
        full_shortest_model = My_ShortestPathModel()
        # evaluate
        predmodel.eval()
        cost_pred_arr = []
        # load data
        for data in dataloader:
            x, c, w, z = data
            # cuda
            if next(predmodel.parameters()).is_cuda:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # predict
            with torch.no_grad(): # no grad
                cp = predmodel(x).to("cpu").detach().numpy()
            # print("cp[0] = ",cp[0])
            # solve
            for j in range(cp.shape[0]):
                sol_pred = full_shortest_model.solve_Shortest_Path(arcs,cp[j],grid)
                cost_pred = np.dot(sol_pred, c[j].to("cpu").detach().numpy())
                cost_pred_arr.append(cost_pred)
                # cost_true_arr.append(z[j].item())
        # turn back train mode
        predmodel.train()
        # normalized
        return cost_pred_arr

    # from pyepo import EPO
    def evaluation_True(self, dataloader):
        cost_true_arr = []
        # load data
        for data in dataloader:
            x, c, w, z = data
            data_size = np.shape(x)[0]
            for j in range(data_size):
                cost_true_arr.append(z[j].item())
        # normalized
        return cost_true_arr

    def obtain_SPO_Cost(self,num_feat,grid,spop,num_epochs,arcs,loader_train,loader_test):
        import time
        import torch
        # init model
        reg = LinearRegression(num_feat,grid)
        # cuda
        if torch.cuda.is_available():
            reg = reg.cuda()
        # set adam optimizer
        optimizer = torch.optim.Adam(reg.parameters(), lr=1e-2)
        # train mode
        reg.train()

        # cost_pred_arr = self.evaluation_SPO(reg, loader_test,arcs,grid)
        cost_pred_arr = 1000000000
        # print("epoch 0: Average SPO Cost = ", np.mean(cost_pred_arr))
        # init elpased time
        elapsed = 0
        for epoch in range(num_epochs):
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
                loss = spop(cp, c, w, z)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record time
                tock = time.time()
                elapsed += tock - tick
                # log
            cost_pred_arr = self.evaluation_SPO(reg, loader_test,arcs,grid)
            # print("epoch = ",epoch," Average SPO Cost = ", np.mean(cost_pred_arr))
        return cost_pred_arr

    def run(self,DataPath_seed,x_train,c_train,x_test,c_test,batch_size,num_feat,grid,num_epochs,is_run_SPO):
        # print("Hello world")
        # 在这里实例化 shortestpathModel
        optmodel = shortestPathModel()
        arcs = optmodel.arcs

        # get optDataset
        dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
        dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

        # set data loader
        from torch.utils.data import DataLoader
        # batch_size = 20
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        # Obtain oracle cost
        cost_Oracle = self.evaluation_True(loader_test)
        # print("oracle cost = ",np.nanmean(cost_Oracle))
        # Obtain SPO cost
        if is_run_SPO:
            spop = pyepo.func.SPOPlus(optmodel, processes=2)
            cost_SPO = self.obtain_SPO_Cost(num_feat,grid,spop,num_epochs,arcs,loader_train,loader_test)
        else:
            cost_SPO = 0

        rst_Oracel = {"cost":cost_Oracle}
        rst_SPO = {"cost":cost_SPO}

        with open(DataPath_seed +'rst_Oracel.pkl', "wb") as tf:
            pickle.dump(rst_Oracel,tf)
        with open(DataPath_seed +'rst_SPO.pkl', "wb") as tf:
            pickle.dump(rst_SPO,tf)
        return arcs,loader_train,loader_test,cost_Oracle,cost_SPO