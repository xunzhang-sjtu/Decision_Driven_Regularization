import numpy as np
import pickle

def set_Params(params_dir):
    Params = {
        'num_train': 100,
        'num_feat': 5,
        'num_test': 1000,
        'deg': 1.0, # polynomial degree
        'e': 0.5, # scale of normal std or the range of uniform. For the error term
        'lower': 0, # coef lower bound
        'upper': 1, # coef upper bound
        'p': 5, # num of features
        'alpha': 0.5, # scale of normal std or the range of uniform. For the error term
        'mis': 1.0, # model misspecification
        'coef_seed': 1,
        'x_dist': 'uniform',
        'e_dist': 'normal',
        'x_low': -2,
        'x_up': 2,
        'x_mean': 2,
        'x_var': 2,
        'bump': 100,
        # 'grid_all':[(2,2),(3,3),(4,4),(5,5)],
        'grid_all':[(3,3)],
        'iteration_all': np.arange(0,100),
        'mu_all':[0.75],
        'lamb_all':[0.8],
    }
    with open(params_dir+'Params.pkl', "wb") as tf:
        pickle.dump(Params,tf)

def get_Params(params_dir):
    with open(params_dir+'Params.pkl', "rb") as tf:
        Params = pickle.load(tf)
    num_train = Params['num_train']
    num_feat = Params['num_feat']
    num_test = Params['num_test']
    deg = Params['deg']  # polynomial degree
    e = Params['e']  # scale of normal std or the range of uniform. For the error term
    lower = Params['lower']  # coef lower bound
    upper = Params['upper']  # coef upper bound
    p = Params['p']  # num of features
    alpha = Params['alpha']  # scale of normal std or the range of uniform. For the error term
    mis = Params['mis']  # model misspecification
    coef_seed = Params['coef_seed']
    x_dist = Params['x_dist']
    e_dist = Params['e_dist']
    x_low = Params['x_low']
    x_up = Params['x_up']
    x_mean = Params['x_mean']
    x_var = Params['x_var']
    bump = Params['bump']
    grid_all = Params['grid_all']
    iteration_all = Params['iteration_all']
    mu_all = Params['mu_all']
    lamb_all = Params['lamb_all']
    return num_train, num_feat, num_test, deg, e, lower, upper, p, alpha, mis, coef_seed, x_dist, e_dist, x_low, x_up, x_mean, x_var, bump, grid_all, iteration_all,mu_all, lamb_all

