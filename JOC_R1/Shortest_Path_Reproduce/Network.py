class network_design:
    def __init__(self):
        pass

    def _getArcs(self,grid):
        arcs = []
        for i in range(grid[0]):
            # edges on rows
            for j in range(grid[1] - 1):
                v = i * grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == grid[0] - 1:
                continue
            for j in range(grid[1]):
                v = i * grid[1] + j
                arcs.append((v, v + grid[1]))

        arc_index_mapping = {}
        for i in range(len(arcs)):
            arc = arcs[i]
            arc_index_mapping[arc] = i

        return arcs,arc_index_mapping