import numpy as np
import copy 
import pathlib
from gurobipy import *
from Optimization_Models import shortestPathModel
import os
import pickle
import out_of_sample_perf as perf
import Oracle

from Data import data_generation
data_gen = data_generation()
np.random.seed(1)

def generate_samples(file_path,p,d,num_test, num_train, alpha, e_dist, tree,feat_names,seed):

    # if os.path.exists(file_path+'Data.pkl'):
    if False:

        with open(file_path+'Data.pkl', "rb") as tf:
            Data = pickle.load(tf)

        x_train = Data["x_train"]
        c_train_avg = Data["c_train_avg"]
        c_train_real = Data["c_train_real"]
        x_test = Data["x_test"]
        c_test_avg = Data["c_test_avg"]
        c_test_real = Data["c_test_real"]
    else:
        # np.random.seed(seed)
        n = num_train+num_test
        c_avg = np.zeros((n, d))
        c_real = np.zeros((n, d))
        x = np.zeros((n,p))
        eps_all = np.zeros((n,d))
        if e_dist == 'normal':
            eps_all = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = n)
        elif e_dist == 'uniform':
            eps_all = np.random.uniform(-alpha,alpha, size = (n, d))

        for i in range(n):
            sample = {}
            f_index = 0
            for feat_name in feat_names:
                x[i,f_index] = np.random.uniform(0, 1)
                sample[feat_name] = x[i,f_index]
                f_index = f_index + 1
            c_avg[i,:] = tree.evaluate(sample)
            c_real[i,:] = c_avg[i,:] + eps_all[i,:]

        x_train = x[0:num_train,:]
        c_train_avg = c_avg[0:num_train,:]
        c_train_real = c_real[0:num_train,:]
        x_test = x[num_train:n,:]
        c_test_avg = c_avg[num_train:n,:]
        c_test_real = c_real[num_train:n,:]

        # dict = {}
        # dict["x_train"] = x_train
        # dict["c_train_avg"] = c_train_avg
        # dict["c_train_real"] = c_train_real
        # dict["x_test"] = x_test
        # dict["c_test_avg"] = c_test_avg
        # dict["c_test_real"] = c_test_real
        # with open(file_path+'Data.pkl', "wb") as tf:
        #     pickle.dump(dict,tf)
    return x_train, c_train_avg, c_train_real, x_test, c_test_avg,c_test_real


class Loss:
    def __init__(self,criteria):
        self.criteria = criteria
        # pass

    def MSEloss(self,C,Cpred):
      #return distance.cdist(C, Cpred, 'sqeuclidean').reshape(-1)
      MSE = (C**2).sum(axis=1)[:, None] - 2 * C.dot(Cpred.transpose()) + ((Cpred**2).sum(axis=1)[None, :])
      return np.sum(MSE.reshape(-1))

    def MSE_Loss(self,Y):
        [rows,cols] = np.shape(Y)
        model = Model('shortest_path')
        model.Params.OutputFlag = 0
        y = model.addVars(cols,lb=-100000, name = 'y')
        err = []
        for i in range(rows):
            for j in range(cols):
                err.append(Y[i,j] - y[j])
        model.setObjective(quicksum([err[k] * err[k] for k in range(len(err))]), GRB.MINIMIZE)
        model.optimize()
        return model.objVal

    def SPOloss(self,SPM,C_this,A_this):
        c_mean = np.mean(C_this,axis = 0)
        rst = SPM.solve_shortest_path(c_mean)
        sol = rst["weights"]
        Edge_dict = SPM.Edge_dict
        Edges = SPM.Edges
        [rows,cols] = np.shape(C_this)
        loss = []
        for k in range(rows):
            c_curr = C_this[k]
            loss.append(sum([(c_curr[Edge_dict[edge]]* sol[edge] ) for edge in Edges]) - A_this[k])
        return np.mean(loss)

    def DDRloss(self,SPM,C_this):
        rst = SPM.solve_DDR_Model(C_this)
        return rst["obj"]


class TreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # only used for leaf

    def is_leaf_node(self):
        return self.left is None and self.right is None

class CARTRegressor:
    def __init__(self, mu,lamb,max_depth=3, min_samples_split=20):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mu = mu
        self.lamb = lamb

    def fit(self,dim,criteria,feat_lb, feat_ub,X, y):
        SPM = shortestPathModel(dim,self.mu,self.lamb)
        if criteria == "SPO":
            num_obs = X.shape[0]
            A = np.zeros(num_obs)
            for i in range(num_obs):
                rst = SPM.solve_shortest_path(y[i])
                A[i] = rst['objective']
            self.root = self._build_tree(SPM,criteria,feat_lb, feat_ub, A,X, y, depth=0)
        else:
            self.root = self._build_tree(SPM,criteria,feat_lb, feat_ub, np.zeros(X.shape[0]),X, y, depth=0)

    def _build_tree(self, SPM,criteria,feat_lb, feat_ub, A,X, y, depth):
        node = TreeNode(depth=depth, max_depth=self.max_depth)
        loss = Loss(criteria)

        # stopping criteria
        # if (depth >= self.max_depth) or (len(y) < self.min_samples_split) or np.var(y) == 0:
        if (depth >= self.max_depth) or np.var(y) == 0:
            if criteria == "MSE":
                node.value = np.mean(y,axis = 0)
            if criteria == "SPO":
                node.value = np.mean(y,axis = 0)
            if criteria == "DDR":
                ddr_rst = SPM.solve_DDR_Model(y)
                node.value = ddr_rst["sol"]
            return node

        best_feature, best_threshold, best_score = None, None, float('inf')
        n_samples, n_features = X.shape

        # try all features and thresholds
        for feature in range(n_features):
            # thresholds = np.unique(X[:, feature])
            thresholds = np.linspace(feat_lb[feature], feat_ub[feature], num=100)

            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                # if len(y[left_indices]) == 0 or len(y[right_indices]) == 0: ## 增加最小数量要求
                if len(y[left_indices]) < self.min_samples_split or len(y[right_indices]) < self.min_samples_split: ## 增加最小数量要求
                    continue

                # calculate MSE
                if criteria == "MSE":
                    loss_left = loss.MSEloss(y[left_indices],np.mean(y[left_indices],axis = 0).reshape(1,-1))
                    loss_right = loss.MSEloss(y[right_indices],np.mean(y[right_indices],axis = 0).reshape(1,-1))
                    metric_value = loss_left + loss_right
                if criteria == "SPO":
                    loss_left = loss.SPOloss(SPM,y[left_indices],A[left_indices])
                    loss_right = loss.SPOloss(SPM,y[right_indices],A[right_indices])
                    metric_value = loss_left + loss_right
                if criteria == "DDR":
                    loss_left = loss.DDRloss(SPM,y[left_indices])
                    loss_right = loss.DDRloss(SPM,y[right_indices])
                    metric_value = loss_left + loss_right

                if metric_value < best_score:
                    best_score = metric_value
                    best_feature = feature
                    best_threshold = threshold

        # if no split improves score, make leaf
        if best_feature is None:
            node.value = np.mean(y)
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.feat_lb = feat_lb
        node.feat_ub = feat_ub
        
        # split and recurse
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        # print("depth=",depth,"feature=",best_feature,"threshold=",best_threshold)
        # print("left_indices=",len(left_indices[left_indices==True]))
        # print("right_indices=",len(right_indices[right_indices==True]))

        feat_lb_left = copy.deepcopy(feat_lb)
        feat_ub_left = copy.deepcopy(feat_ub)
        feat_ub_left[best_feature] = best_threshold
        node.left = self._build_tree(SPM,criteria,feat_lb_left,feat_ub_left,A[left_indices],X[left_indices], y[left_indices], depth + 1)

        feat_lb_right = copy.deepcopy(feat_lb)
        feat_ub_right = copy.deepcopy(feat_ub)
        feat_lb_right[best_feature] = best_threshold
        node.right = self._build_tree(SPM,criteria,feat_lb_right,feat_ub_right,A[right_indices],X[right_indices], y[right_indices], depth + 1)

        return node

    def predict_one(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X,num_routs):
        [rows,cols] = np.shape(X)
        pred = np.zeros((rows,num_routs))
        for row in range(rows):
            pred[row,:] = self.predict_one(X[row,:], self.root)
        return pred
        # return np.array([self.predict_one(x, self.root) for x in X])

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.is_leaf_node():
            print(indent + "Leaf: Predict = ",np.round(node.value,3))
        else:
            print(indent + f"[x{node.feature_index} < {node.threshold:.3f}]")
            self.print_tree(node.left, indent + "  ")
            self.print_tree(node.right, indent + "  ")



########################################
#problem parameters
dim = 3
d = (dim - 1) * (dim - 1) * 2 + 2 * (dim - 1) # num of arcs
num_train = 100
num_feat = 1 # size of feature
num_test = 1000
e = 100 # scale of normal std or the range of uniform. For the error term
lower = 0 # coef lower bound
upper = 1 # coef upper bound
p = num_feat # num of features
alpha = e # scale of normal std or the range of uniform. For the error term
coef_seed = 1
x_dist = 'uniform'
e_dist = 'normal'
x_low = -2
x_up = 2
x_mean = 2
x_var = 2
bump = 100
deg = 2.0
mis = deg


reps_st = 0 #0 #can be as low as 0
reps_end = 100 #1 #can be as high as 50
iteration_all = np.arange(reps_st,reps_end)
#training parameters
max_depth = 3 #3
max_depth_true = 3
min_weights_per_node = 20 #20

mu_all = np.round(np.arange(0.1,1,0.1),4)
mu_all = [0.5]
lamb_all = np.append(np.append(np.round(np.arange(0.1,1.0,0.2),4),np.arange(1.0,10.0,2.0)),np.arange(10,100,20))
# lamb_all = np.append(lamb_all,np.arange(100,1000,200))
lamb_all = np.arange(100,1000,50)
# lamb_all = [0.5]
########################################
data_generation_process = "Tree_based_Data_Generation"
# data_generation_process = "SPO_Data_Generation"
current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
DataPath_parent = grandparent_directory + '/Data_JOC_R1/Shortest_Path_Tree/dim='+str(dim) +'_depth_'+str(max_depth)+"_Tree_based_Data_Generation/"
print("DataPath_parent:", DataPath_parent)
result_dir = DataPath_parent +"result/Data_size="+str(num_train)+"/"
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
print("result_dir:", result_dir)
DataPath = DataPath_parent + "data_size="+str(num_train)+"_e="+str(e)+"_p="+str(p)+"_x_dist="+x_dist+"_num_test="+str(num_test)+"/"
pathlib.Path(DataPath).mkdir(parents=True, exist_ok=True)


feat_lb=np.zeros(p); feat_ub=np.ones(p)
cost_Oracle = {}; cost_MSE = {}; cost_SPO = {}; cost_DDR = {}; 

for iter in iteration_all:

    if data_generation_process == "Tree_based_Data_Generation":
        feat_names = [f"x_{i}" for i in range(num_feat)]
        from Data import VectorBinaryTree
        tree = VectorBinaryTree(feat_lb=feat_lb,feat_ub=feat_ub,depth=max_depth_true, max_depth=max_depth, output_dim=d, current_depth=0, feature_names=feat_names)
        x_train, c_train_ante, c_train_post, x_test, c_test_ante, c_test_post = generate_samples(DataPath,p,d,num_test, num_train, alpha, e_dist, tree,feat_names,1)
        # print("x_train = ",x_train[0,:])
    if data_generation_process == "SPO_Data_Generation":
        DataPath_iter = DataPath +"iter="+str(iter)+"/"
        pathlib.Path(DataPath_iter).mkdir(parents=True, exist_ok=True)
        W_star = data_gen.generate_truth(DataPath_iter,lower, upper, p, d, iter,data_generation_process) 
        # #  ****** Data generation *********
        x_train, c_train_ante, c_train_post, x_test, c_test_ante, c_test_post = data_gen.generate_samples(iter,DataPath_iter,p, d, num_test, num_train, alpha, W_star, mis, num_test, 
                                data_generation_process, x_dist, e_dist, x_low, x_up, x_mean, x_var, bump) 

    ### Run Oracle ###
    cost_Oracle[iter] = perf.compute_oof_cost(c_test_ante,c_test_ante,dim,0,0)
    print("iter = ",iter, ",Oracle avg cost = ",np.mean(cost_Oracle[iter]))

    ### Run MSE Tree ###
    MSE_Model = CARTRegressor(0,0,max_depth=max_depth)
    MSE_Model.fit(dim,"MSE",feat_lb,feat_ub,x_train, c_train_post)
    cost_MSE_Pred = MSE_Model.predict(x_test,d)
    cost_MSE[iter] = perf.compute_oof_cost(cost_MSE_Pred,c_test_ante,dim,0,0)
    print("iter = ",iter, ",MSE avg cost = ",np.mean(cost_MSE[iter]))

    ### Run SPO Tree ###
    SPO_Model = CARTRegressor(0,0,max_depth=max_depth)
    SPO_Model.fit(dim,"SPO",feat_lb,feat_ub,x_train, c_train_post)
    cost_SPO_Pred = SPO_Model.predict(x_test,d)
    cost_SPO[iter] = perf.compute_oof_cost(cost_SPO_Pred,c_test_ante,dim,0,0)
    print("iter = ",iter, ",SPO avg cost = ",np.mean(cost_SPO[iter]))


    # ### Run DDR Tree ###
    for mu in mu_all:
        for lamb in lamb_all:
            DDR_Model = CARTRegressor(mu,lamb,max_depth=max_depth)
            DDR_Model.fit(dim,"DDR",feat_lb,feat_ub,x_train, c_train_post)
            cost_DDR_Pred = DDR_Model.predict(x_test,d)
            cost_DDR[iter,mu,lamb] = perf.compute_oof_cost(cost_DDR_Pred,c_test_ante,dim,0,0)
            # print("iter = ",iter,", mu=",mu,", lamb=",lamb, ",DDR avg cost = ",np.mean(cost_DDR[iter,mu,lamb]))
            print("iter = ",iter, ",mu=",mu,",lamb=",lamb,\
                ",DDR vs SPO cost = ",np.round(np.mean(cost_DDR[iter,mu,lamb])/np.mean(cost_SPO[iter]),4),\
                ",DDR vs MSE cost = ",np.round(np.mean(cost_DDR[iter,mu,lamb])/np.mean(cost_MSE[iter]),4))

print("result_dir:", result_dir)
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
with open(result_dir+'cost_Oracle.pkl', "wb") as tf:
    pickle.dump(cost_Oracle,tf)
with open(result_dir+'cost_MSE.pkl', "wb") as tf:
    pickle.dump(cost_MSE,tf)
with open(result_dir+'cost_SPO.pkl', "wb") as tf:
    pickle.dump(cost_SPO,tf)
with open(result_dir+'cost_DDR.pkl', "wb") as tf:
    pickle.dump(cost_DDR,tf)

# # 测试预测
# y_pred = model.predict(X[:5])
# print("Predictions:", y_pred)
