import numpy as np
from scipy.stats import bernoulli
import pickle

class data_generation:
    def __init__(self):
        pass

    def generate_truth(self,lower,upper,p,d, version):
        # p number of feature; 
        # d number of routes
        # B_star is w in the paper
        # w ~ U[lower, upper]
        # version "DDR_Data_Generation": uniformly distributed in [lower, upper]
        # version 1: each component is bernoulli with prob == 0.5
        # version 2: W* = (W^0,0)
        # version 3: W* = (W sparse, 0)
        #     random.seed(seed)
        if version == "DDR_Data_Generation":
            W_star = np.random.uniform(lower,upper,(d,p))
        elif version == 1:
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

        return W_star
        

    def generate_samples(self,file_path,p,d,samples_test, samples_train, alpha, W_star, n_epsilon, mis, thres, 
                        version, x_dist, e_dist, x_low, x_up, x_mean, x_var, bump):
        # upper and lower are not used
        # mis is the beta in the paper
        if version == "DDR_Data_Generation": ## X ~ U[x_low,x_up]; epsilon ~ N(0,alpha)
            if x_dist == 'normal':
    #             x_test= np.random.uniform(x_low, x_up, size = (samples_test,p))
    #             x_train = np.random.uniform(x_low, x_up, size = (samples_train,p))
                x_test= np.random.multivariate_normal(x_mean*np.ones(p), x_var*np.identity(p), size = samples_test) ## KK
                x_train = np.random.multivariate_normal(x_mean*np.ones(p), x_var*np.identity(p), size = samples_train) ## KK
                
            elif x_dist == 'uniform':
                x_test= np.random.uniform(x_low, x_up, size = (samples_test,p))
                x_train = np.random.uniform(x_low, x_up, size = (samples_train,p))
            
            z_test_ori = np.power(np.dot(x_test, W_star.T) + bump * np.ones((samples_test, d)), mis)
            z_train_ori = np.power(np.dot(x_train, W_star.T) + bump * np.ones((samples_train, d)), mis)
            
            if e_dist == 'normal':
                noise_test = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = samples_test)
                noise_train = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = samples_train)
            elif e_dist == 'uniform':
                noise_test = np.random.uniform(-alpha,alpha, size = (samples_test, d))
    #             noise_train = np.random.uniform(x_low, x_up, size = (samples_train, p))  #### why x_low and x_up
                noise_train = np.random.uniform(-alpha, alpha, size = (samples_train, d))  ## KK

            z_test = z_test_ori + noise_test
            z_train = z_train_ori + noise_train
        
        elif version == 2: # n_epsilon
            if x_dist == 'normal':
                x_test= np.random.multivariate_normal(x_mean*np.zeros(p), x_var*np.identity(p), size = samples_test)
                x_train = np.random.multivariate_normal(x_mean*np.zeros(p), x_var*np.identity(p), size = samples_train)
            elif x_dist == 'uniform':
                x_test= np.random.uniform(x_low, x_up, size = (samples_test,p))
                x_train = np.random.uniform(x_low, x_up, size = (samples_train,p))
            
            z_test_ori = np.power(np.dot(x_test, W_star.T) + bump * np.ones((samples_test, d)), mis)
            z_train_ori = np.power(np.dot(x_train, W_star.T) + bump * np.ones((samples_train, d)), mis)
            
            if e_dist == 'normal':
                z_test = [z_test_ori + np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = samples_test) for n in range(n_epsilon)]
                noise_train = np.random.multivariate_normal(np.zeros(d), alpha*np.identity(d), size = samples_train)
            elif e_dist == 'uniform':
                z_test = [z_test_ori + np.random.uniform(-alpha, alpha, size = (samples_test,d)) for n in range(n_epsilon)]
                noise_train = np.random.uniform(-alpha, alpha, size = (samples_train, d))  ## KK
            
            z_train = z_train_ori + noise_train
            
        elif version == 3: ## SPO+ version
            x_test= np.random.multivariate_normal(np.zeros(p), np.identity(p), size = samples_test)
            hold = np.dot(x_test, W_star.T)/np.sqrt(p) + 3
            sign_hold = np.sign(hold)
            z_test_ori = np.multiply(sign_hold, np.power(np.abs(hold), mis)) + 1
            noise = np.random.uniform(1 - alpha, 1 + alpha, size = (samples_test,d))
            z_test =np.multiply(z_test_ori, noise) 

            x_train=np.random.multivariate_normal(np.zeros(p), np.identity(p), size = samples_train)
            hold = np.dot(x_train, W_star.T)/np.sqrt(p) + 3
            sign_hold = np.sign(hold)
            z_train_ori = np.multiply(sign_hold, np.power(np.abs(hold), mis)) + 1
            noise = np.random.uniform(1 - alpha,1 + alpha, size = (samples_train, d))
            z_train = np.multiply(z_train_ori, noise) 
        

        dict = {}
        dict["x_test"] = x_test
        dict["z_test_ori"] = z_test_ori
        dict["z_test"] = z_test
        dict["x_train"] = x_train
        dict["z_train_ori"] = z_train_ori
        dict["z_train"] = z_train
        dict["W_star"] = W_star
        with open(file_path+'Data.pkl', "wb") as tf:
            pickle.dump(dict,tf)

        return x_test, z_test_ori, z_test, x_train, z_train_ori, z_train, W_star
