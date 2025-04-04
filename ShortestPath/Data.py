import numpy as np
from scipy.stats import bernoulli
import pickle
import os

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
        # np.random.seed(seed)
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
        # upper and lower are not used
        # mis is the beta in the paper
        if os.path.exists(file_path+'Data.pkl'):
            with open(file_path+'Data.pkl', "rb") as tf:
                Data = pickle.load(tf)
            x_test = Data["x_test"]
            c_test = Data["c_test"]
            x_train = Data["x_train"]
            c_train = Data["c_train"]
            W_star = Data["W_star"]
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
                c = np.zeros((n, d))
                x = np.zeros((n,p))
                for i in range(n):
                    c_tem = np.ones(p) * -1
                    while np.min(c_tem)< 0:
                        # feature vector
                        # xi = rnd.normal(0,1,p)
                        xi = np.random.normal(0,1,p)
                        # cost without noise
                        c_tem = (np.dot(W_star, xi.reshape(p, 1)).T / np.sqrt(p) + 3) 
                    ci = c_tem ** mis + 1
                    # # rescale
                    # ci /= 3.5 ** mis
                    # noise
                    epislon = np.random.uniform(1 - alpha, 1 + alpha, d)
                    ci *= epislon

                    x[i,:] = xi
                    c[i, :] = ci

                from sklearn.model_selection import train_test_split
                x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=num_test, random_state=42)


            dict = {}
            dict["x_test"] = x_test
            dict["c_test"] = c_test
            dict["x_train"] = x_train
            dict["c_train"] = c_train
            dict["W_star"] = W_star
            with open(file_path+'Data.pkl', "wb") as tf:
                pickle.dump(dict,tf)

        # print("W_star = ",W_star[0,:])
        # print("x_train = ",x_train[0,:])
        # print("z_train = ",z_train[0,:])
        return x_test, c_test, x_train, c_train, W_star


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



