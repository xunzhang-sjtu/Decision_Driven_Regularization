import numpy as np
from scipy.stats import bernoulli
import pickle
import os
import pandas as pd
import copy

class data_generation:
    def __init__(self):
        pass

    def generate_truth(self,file_path,lower,upper,p,d, seed,version):
        # p number of feature; 
        # d number of routes
        # B_star is w in the paper
        # w ~ U[lower, upper]
        # version 0: uniformly distributed in [lower, upper]
        # version 1: each component is bernoulli with prob == 0.5
        # version 2: W* = (W^0,0)
        # version 3: W* = (W sparse, 0)
        np.random.seed(seed)
        if version == "DDR_Data_Generation":
            W_star = np.random.uniform(lower,upper,(d,p))
        elif version == "SPO_Data_Generation":
            prob = 0.5
            X = bernoulli(prob)
            W_star = X.rvs((d,p))
        elif version == 2:
            W_star = np.concatenate((np.random.uniform(lower,upper,(d,np.int(0.75*p))), np.zeros((d, p - np.int(0.75*p)))), axis = 1)
        elif version == 3:
            prob = 0.5
            X = bernoulli(prob)
            W_star = np.concatenate( ( X.rvs((d,np.int(0.75*p))), np.zeros((d, p - np.int(0.75*p))) ), axis = 1 )
        elif version == 4:
            W_U = np.random.uniform(lower,upper,(d,p))
            prob = 0.5
            X = bernoulli(prob)
            W_B = X.rvs((d,p))
            W_star = np.multiply(W_U, W_B)
        elif version == 5: ## 0(0.5),1(0.25),2(0.25)
            W_star = np.random.choice([0,1,2], (d,p), p =[0.5,0.25,0.25])
        if version == 6:
            W_star = np.random.uniform(lower,upper,(d,p))
            W_star_same = np.random.uniform(lower,upper,p)
            for d_index in range(d):
                W_star[d_index,:] = W_star_same + np.random.uniform(lower,upper,p) * 0.1
        # dict = {}
        # dict["W_star"] = W_star
        # with open(file_path+'W_star.pkl', "wb") as tf:
        #     pickle.dump(W_star,tf)
        return W_star
        

    def generate_samples(self,iter,file_path,p,d,num_test, num_train, alpha, W_star, mis, thres, 
                        version, x_dist, e_dist, x_low, x_up, x_mean, 
                        x_var, bump):
        # if os.path.exists(file_path+'Data.pkl'):
        if False:
            with open(file_path+'Data.pkl', "rb") as tf:
                Data = pickle.load(tf)
            x_test = Data["x_test"]
            c_test = Data["c_test"]
            x_train = Data["x_train"]
            c_train = Data["c_train"]
            W_star = Data["W_star"]
            noise_train = Data["noise_train"]
            noise_test = Data["noise_test"]
        else:
            # np.random.seed(iter)
            if version == "DDR_Data_Generation": ## X ~ U[x_low,x_up]; epsilon ~ N(0,alpha)
                if x_dist == 'normal':
                    # x_test= np.random.uniform(x_low, x_up, size = (samples_test,p))
                    # x_train = np.random.uniform(x_low, x_up, size = (samples_train,p))
                    x_test= np.random.multivariate_normal(x_mean*np.ones(p), x_var*np.identity(p), size = num_test) ## KK
                    x_train = np.random.multivariate_normal(x_mean*np.ones(p), x_var*np.identity(p), size = num_train) ## KK
                    
                elif x_dist == 'uniform':
                    x_test= np.random.uniform(x_low, x_up, size = (num_test,p))
                    x_train = np.random.uniform(x_low, x_up, size = (num_train,p))
                
                z_test_ori = np.power(np.dot(x_test, W_star.T) + bump * np.ones((num_test, d)), mis)
                z_train_ori = np.power(np.dot(x_train, W_star.T) + bump * np.ones((num_train, d)), mis)
                
                if e_dist == 'normal':
                    noise_test = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = num_test)
                    noise_train = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = num_train)
                elif e_dist == 'uniform':
                    noise_test = np.random.uniform(-alpha,alpha, size = (num_test, d))
        #             noise_train = np.random.uniform(x_low, x_up, size = (samples_train, p))  #### why x_low and x_up
                    noise_train = np.random.uniform(-alpha, alpha, size = (num_train, d))  ## KK

                c_test = z_test_ori + noise_test
                c_train = z_train_ori + noise_train
            
            elif version == 2: # n_epsilon
                if x_dist == 'normal':
                    x_test= np.random.multivariate_normal(x_mean*np.zeros(p), x_var*np.identity(p), size = num_test)
                    x_train = np.random.multivariate_normal(x_mean*np.zeros(p), x_var*np.identity(p), size = num_train)
                elif x_dist == 'uniform':
                    x_test= np.random.uniform(x_low, x_up, size = (num_test,p))
                    x_train = np.random.uniform(x_low, x_up, size = (num_train,p))
                
                z_test_ori = np.power(np.dot(x_test, W_star.T) + bump * np.ones((num_test, d)), mis)
                z_train_ori = np.power(np.dot(x_train, W_star.T) + bump * np.ones((num_train, d)), mis)
                
                if e_dist == 'normal':
                    # z_test = [z_test_ori + np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = samples_test) for n in range(n_epsilon)]
                    z_test = z_test_ori + np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = num_test)

                    noise_train = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = num_train)
                elif e_dist == 'uniform':
                    z_test = z_test_ori + np.random.uniform(-alpha, alpha, size = (num_test,d)) 
                    noise_train = np.random.uniform(-alpha, alpha, size = (num_train, d))  ## KK
                
                z_train = z_train_ori + noise_train
                
            elif version == "SPO_Data_Generation": ## SPO+ version
                if mis <= 0:
                    raise ValueError("deg = {} should be positive.".format(mis))
                # set seed
                # rnd = np.random.RandomState(seed)
                n = num_train+num_test
                c_ante = np.zeros((n, d))
                c_post = np.zeros((n, d))
                x = np.zeros((n,p))
                eps_all = np.zeros((n,d))
                for i in range(n):
                    c_tem = np.ones(p) * -1
                    while np.min(c_tem)< 0:
                        # feature vector
                        # xi = rnd.normal(0,1,p)
                        xi = np.random.normal(0,1,p)
                        # cost without noise
                        c_tem = (np.dot(W_star, xi.reshape(p, 1)).T / np.sqrt(p) + 1) 
                    ci = c_tem ** mis
                    # # rescale
                    # ci /= 3.5 ** mis
                    # noise
                    epislon = np.random.uniform(1 - alpha, 1 + alpha, d)
                    ci_post = ci * epislon

                    x[i,:] = xi
                    c_ante[i,:] = ci
                    eps_all[i,:] = epislon
                    c_post[i,:] = ci_post
                x_train = x[0:num_train,:]
                x_test = x[num_train:n,:]
                c_train_ante = c_ante[0:num_train,:]
                c_test_ante = c_ante[num_train:n,:]
                c_train_post = c_post[0:num_train,:]
                c_test_post = c_post[num_train:n,:]

            # dict = {}
            # dict["x_test"] = x_test
            # dict["c_test"] = c_test
            # dict["x_train"] = x_train
            # dict["c_train"] = c_train
            # dict["noise_train"] = noise_train
            # dict["noise_test"] = noise_test
            # dict["W_star"] = W_star
            # with open(file_path+'Data.pkl', "wb") as tf:
            #     pickle.dump(dict,tf)

        # print("W_star = ",W_star[0,:])
        # print("x_train = ",x_train[0,:])
        # print("z_train = ",z_train[0,:])
        return x_train, c_train_ante, c_train_post, x_test, c_test_ante, c_test_post
        # return x_test, c_test, x_train, c_train, noise_train,noise_test,W_star


    def generate_SPO_Data_True_Coef(self,coef_seed,grid,num_data,num_features):
        rnd = np.random.RandomState(coef_seed)
        # numbrnda points
        n = num_data
        # dimension of features
        p = num_features
        # dimension of the cost vector
        d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (d, p))
        return B

    def generate_Shortest_Path_Data(self,num_data, num_features, grid, deg=1, noise_width=0, seed=135,coef_seed=1):
        """
        A function to generate synthetic data and features for shortest path

        Args:
            num_data (int): number of data points
            num_features (int): dimension of features
            grid (int, int): size of grid network
            deg (int): data polynomial degree
            noise_width (float): half witdth of data random noise
            seed (int): random seed

        Returns:
        tuple: data features (np.ndarray), costs (np.ndarray)
        """
        # # positive integer parameter
        # if type(deg) is not int:
        #     raise ValueError("deg = {} should be int.".format(deg))
        if deg <= 0:
            raise ValueError("deg = {} should be positive.".format(deg))
        # set seed
        rnd = np.random.RandomState(seed)
        # numbrnda points
        n = num_data
        # dimension of features
        p = num_features
        # dimension of the cost vector
        d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
        # random matrix parameter B
        # B = rnd.binomial(1, 0.5, (d, p))
        B = self.generate_SPO_Data_True_Coef(coef_seed,grid,num_data,num_features)
        print("B = ",B[:,1])
        # cost vectors
        c = np.zeros((n, d))
        x = np.zeros((n,p))
        for i in range(n):
            c_tem = np.ones(p) * -1
            while np.min(c_tem)< 0:
                # feature vector
                xi = rnd.normal(0,1,p)
                # cost without noise
                c_tem = (np.dot(B, xi.reshape(p, 1)).T / np.sqrt(p) + 3) 
            ci = c_tem ** deg + 1
            # rescale
            ci /= 3.5 ** deg
            # noise
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, d)
            ci *= epislon

            x[i,:] = xi
            c[i, :] = ci

        return x, c




class VectorBinaryTreeNode:
    # np.random.seed(1)
    def __init__(self, depth, max_depth, output_dim, current_depth=0, feature_names=None):
        """
        带向量输出的二叉树节点类
        
        参数:
        - depth: 树的深度
        - max_depth: 最大深度
        - output_dim: 输出向量维度
        - current_depth: 当前深度
        - feature_names: 特征名称列表
        """
        self.depth = current_depth
        self.max_depth = max_depth
        self.output_dim = output_dim
        self.feature_names = feature_names or [f'feature_{i}' for i in range(max_depth)]
        self.is_leaf = current_depth >= max_depth
        
        if not self.is_leaf:
            # 内部节点属性
            self.split_feature = self.feature_names[current_depth]
            self.split_threshold = np.random.uniform(-1, 1)  # 随机分裂阈值
            self.left = VectorBinaryTreeNode(depth, max_depth, output_dim, current_depth+1, feature_names)
            self.right = VectorBinaryTreeNode(depth, max_depth, output_dim, current_depth+1, feature_names)
        else:
            # 叶子节点属性 - 现在是向量
            self.value = np.random.uniform(1000, 2000, size=output_dim)  # 随机向量估计值
    
    def evaluate(self, x):
        """评估样本x，返回向量"""
        if self.is_leaf:
            return self.value
        
        if x[self.split_feature] <= self.split_threshold:
            return self.left.evaluate(x)
        else:
            return self.right.evaluate(x)
    
    def get_rules(self, parent_rules=None):
        """获取从根节点到当前节点的规则和向量值"""
        if parent_rules is None:
            parent_rules = []
            
        if self.is_leaf:
            return [{'rules': parent_rules, 'value': self.value}]
        
        left_rules = parent_rules + [f"{self.split_feature} <= {self.split_threshold:.2f}"]
        right_rules = parent_rules + [f"{self.split_feature} > {self.split_threshold:.2f}"]
        
        return self.left.get_rules(left_rules) + self.right.get_rules(right_rules)

    def generate_vector_binary_tree(self,depth=3, output_dim=2, feature_names=None):
        """
        生成带向量输出的二叉树结构
        
        参数:
        - depth: 树的深度
        - output_dim: 输出向量维度
        - feature_names: 可选的特征名称列表
        
        返回:
        - VectorBinaryTreeNode: 树的根节点
        """
        return self.VectorBinaryTreeNode(depth, depth, output_dim, feature_names=feature_names)

    def generate_samples(self,file_path,p,d,num_test, num_train, alpha, e_dist, tree,feat_names):

        if os.path.exists(file_path+'Data.pkl'):
            with open(file_path+'Data.pkl', "rb") as tf:
                Data = pickle.load(tf)

            x_train = Data["x_train"]
            c_train_avg = Data["c_train_avg"]
            c_train_real = Data["c_train_real"]
            x_test = Data["x_test"]
            c_test_avg = Data["c_test_avg"]
            c_test_real = Data["c_test_real"]
        else:
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
                    x[i,f_index] = np.random.uniform(-1, 1)
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

            dict = {}
            dict["x_train"] = x_train
            dict["c_train_avg"] = c_train_avg
            dict["c_train_real"] = c_train_real
            dict["x_test"] = x_test
            dict["c_test_avg"] = c_test_avg
            dict["c_test_real"] = c_test_real
            with open(file_path+'Data.pkl', "wb") as tf:
                pickle.dump(dict,tf)
        return x_train, c_train_avg, c_train_real, x_test, c_test_avg,c_test_real


class VectorBinaryTree:
    # np.random.seed(1)
    def __init__(self, feat_lb, feat_ub,depth, max_depth, output_dim, current_depth=0, feature_names=None):
        """
        带向量输出的二叉树节点类
        
        参数:
        - depth: 树的深度
        - max_depth: 最大深度
        - output_dim: 输出向量维度
        - current_depth: 当前深度
        - feature_names: 特征名称列表
        """
        self.feat_lb = feat_lb
        self.feat_ub = feat_ub
        self.depth = current_depth
        self.max_depth = max_depth
        self.output_dim = output_dim
        self.feature_names = feature_names or [f'feature_{i}' for i in range(max_depth)]
        self.is_leaf = current_depth >= max_depth
        
        if not self.is_leaf:
            # 内部节点属性
            feat_index = np.random.randint(0, len(self.feat_lb))
            self.split_feature = self.feature_names[feat_index]
            split_val = np.random.uniform(self.feat_lb[feat_index], self.feat_ub[feat_index])
            self.split_threshold = split_val
            feat_ub_left = copy.deepcopy(self.feat_ub)
            feat_ub_left[feat_index] = split_val

            feat_lb_right = copy.deepcopy(self.feat_lb)
            feat_lb_right[feat_index] = split_val
            # self.split_threshold = np.random.uniform(0, 1)  # 随机分裂阈值
            self.left = VectorBinaryTree(self.feat_lb,feat_ub_left,depth, max_depth, output_dim, current_depth+1, feature_names)
            self.right = VectorBinaryTree(feat_ub_left, self.feat_ub,depth, max_depth, output_dim, current_depth+1, feature_names)
        else:
            # 叶子节点属性 - 现在是向量
            self.value = np.random.uniform(1000, 2000, size=output_dim)  # 随机向量估计值
    
    def evaluate(self, x):
        """评估样本x，返回向量"""
        if self.is_leaf:
            return self.value
        
        if x[self.split_feature] <= self.split_threshold:
            return self.left.evaluate(x)
        else:
            return self.right.evaluate(x)
    
    def get_rules(self, parent_rules=None):
        """获取从根节点到当前节点的规则和向量值"""
        if parent_rules is None:
            parent_rules = []
            
        if self.is_leaf:
            return [{'rules': parent_rules, 'value': self.value}]
        
        left_rules = parent_rules + [f"{self.split_feature} <= {self.split_threshold:.2f}"]
        right_rules = parent_rules + [f"{self.split_feature} > {self.split_threshold:.2f}"]
        
        return self.left.get_rules(left_rules) + self.right.get_rules(right_rules)

    def generate_vector_binary_tree(self,depth=3, output_dim=2, feature_names=None):
        """
        生成带向量输出的二叉树结构
        
        参数:
        - depth: 树的深度
        - output_dim: 输出向量维度
        - feature_names: 可选的特征名称列表
        
        返回:
        - VectorBinaryTreeNode: 树的根节点
        """
        return self.VectorBinaryTreeNode(depth, depth, output_dim, feature_names=feature_names)






# # 示例使用
# if __name__ == "__main__":
#     # 1. 生成深度为3，输出维度为3的二叉树
#     feature_names = ['age', 'income']
#     tree = generate_vector_binary_tree(depth=2, output_dim=2, feature_names=feature_names[:3])
    
#     # 2. 打印所有叶子节点的规则和向量值
#     print("二叉树所有路径规则和叶子节点向量值:")
#     for i, rule_info in enumerate(tree.get_rules()):
#         print(f"路径 {i+1}: {' AND '.join(rule_info['rules'])}")
#         print(f"向量值: {rule_info['value']}")
#         print("-" * 50)
    
#     # # 3. 可视化树结构
#     # dot = tree.visualize()
#     # dot.render('vector_binary_tree', format='png', cleanup=True)
#     # print("二叉树可视化已保存为 vector_binary_tree.png")
    
#     # 4. 演示评估样本
#     sample = {'age': 0.5, 'income': -0.8, 'education': 1.2}
#     prediction = tree.evaluate(sample)
#     print(f"\n样本评估: {sample} -> 预测向量: {prediction}")