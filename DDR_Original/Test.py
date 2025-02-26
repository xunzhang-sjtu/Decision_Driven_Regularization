import pyepo
import torch
from torch import nn
from torch.utils.data import DataLoader

# model for shortest path
grid = (5,5) # grid size
optmodel = pyepo.model.grb.shortestPathModel(grid)

# generate data
num_data = 1000 # number of data
num_feat = 5 # size of feature
deg = 4 # polynomial degree
noise_width = 0.5 # noise width
x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

# build dataset
dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

# get data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# build linear model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 40)

    def forward(self, x):
        out = self.linear(x)
        return out
# init
predmodel = LinearRegression()
# set optimizer
optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
# init SPO+ loss
spo = pyepo.func.SPOPlus(optmodel, processes=2)

# training
num_epochs = 2
for epoch in range(num_epochs):
    print("epoch = ",epoch)
    it = 0
    for data in dataloader:
        x, c, w, z = data
        # forward pass
        cp = predmodel(x)
        # SPO+ loss
        loss = spo(cp, c, w, z)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it = ",it,"x = ",spo.optmodel.x)