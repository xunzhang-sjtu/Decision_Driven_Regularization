class My_ShortestPathModel:
    def __init__(self):
        pass
    # arcs_arr = optmodel.arcs
    def obtain_path(self,arcs_arr,sol):
        path_arr = []
        for arc_index in range(len(arcs_arr)):
            if sol[arc_index] > 0:
                path_arr.append(arcs_arr[arc_index])
        return path_arr

    # def getArcs(self,grid):
    #     arcs = []
    #     for i in range(grid[0]):
    #         # edges on rows
    #         for j in range(grid[1] - 1):
    #             v = i * grid[1] + j
    #             arcs.append((v, v + 1))
    #         # edges in columns
    #         if i == grid[0] - 1:
    #             continue
    #         for j in range(grid[1]):
    #             v = i * grid[1] + j
    #             arcs.append((v, v + grid[1]))
    #     return arcs

    def solve_Shortest_Path(self,arcs,cost,grid):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        import gurobipy as gp
        from gurobipy import GRB
        # ceate a model
        m = gp.Model("shortest path")
        m.setParam('OutputFlag', 0)
        # varibles
        x = m.addVars(arcs, name="x")
        # sense
        # m.modelSense = GRB.MINIMIZE
        # flow conservation constraints
        for i in range(grid[0]):
            for j in range(grid[1]):
                v = i * grid[1] + j
                expr = 0
                for e in arcs:
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
                elif i == grid[0] - 1 and j == grid[0] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        m.setObjective( sum([cost[ind] * x[arcs[ind]] for ind in range(len(arcs))]) , GRB.MINIMIZE)
        m.optimize()
        sol = m.getAttr('x')
        # print("sol = ",sol)
        # shortest_path = self.obtain_path(arcs_arr,sol)
        # print("shortest_path = ",shortest_path)
        return sol