import numpy as np
import pickle 

def get_Params(Params_dir):
    with open(Params_dir+'Params.pkl', "rb") as tf:
        Params = pickle.load(tf)
    grid = Params["grid"]
    num_test = Params["num_test"]
    lower = Params["lower"]
    upper = Params["upper"]
    coef_seed = Params["coef_seed"]
    x_dist = Params["x_dist"]
    e_dist = Params["e_dist"]
    x_low = Params["x_low"]
    x_up = Params["x_up"]
    x_mean = Params["x_mean"]
    x_var = Params["x_var"]
    bump = Params["bump"]
    iteration_all = Params["iteration_all"]
    batch_size = Params["batch_size"]
    num_epochs = Params["num_epochs"]
    mu_all = Params["mu_all"]
    lamb_all = Params["lamb_all"]
    return grid,num_test,lower,upper,coef_seed,x_dist,e_dist,x_low,x_up,x_mean,x_var,bump,iteration_all,batch_size,num_epochs,mu_all,lamb_all

def set_Params_Data(Params_dir):
    Params_this = {}
    Params_this["num_feat"] = 5
    Params_this["p"] = Params_this["num_feat"]
    Params_this["deg"] = 1.0
    Params_this["mis"] = Params_this["deg"]
    Params_this["e"] = 0.5
    Params_this["alpha"] = Params_this["e"]
    Params_this["num_train_all"] = [50,100,200,500,1000]

    with open(Params_dir+'Params_Data_Size.pkl', "wb") as tf:
        pickle.dump(Params_this,tf)

def get_Params_Data(Params_dir):
    with open(Params_dir+'Params_Data_Size.pkl', "rb") as tf:
        Params_this = pickle.load(tf)
    num_feat = Params_this["num_feat"]
    p = Params_this["p"]
    deg = Params_this["deg"]
    mis = Params_this["mis"]
    e = Params_this["e"]
    alpha = Params_this["alpha"]
    num_train_all = Params_this["num_train_all"]
    return num_feat,p,deg,mis,e,alpha,num_train_all


def set_Params_Feature(Params_dir):
    Params_this = {}
    Params_this["deg"] = 1.0
    Params_this["mis"] = Params_this["deg"]
    Params_this["e"] = 0.5
    Params_this["alpha"] = Params_this["e"]
    Params_this["num_train"] = 100
    Params_this["num_feat_all"] = [1,3,5,7,10,15]

    with open(Params_dir+'Params_Feature.pkl', "wb") as tf:
        pickle.dump(Params_this,tf)

def get_Params_Feature(Params_dir):
    with open(Params_dir+'Params_Feature.pkl', "rb") as tf:
        Params_this = pickle.load(tf)
    num_feat_all = Params_this["num_feat_all"]
    deg = Params_this["deg"]
    mis = Params_this["mis"]
    e = Params_this["e"]
    alpha = Params_this["alpha"]
    num_train = Params_this["num_train"]
    return num_feat_all,deg,mis,e,alpha,num_train


def set_Params_Noise(Params_dir):
    Params_this = {}
    Params_this["deg"] = 1.0
    Params_this["mis"] = Params_this["deg"]
    Params_this["e_all"] = [0.25,0.5,0.75,1.0]
    # Params_this["alpha"] = Params_this["e"]
    Params_this["num_train"] = 100
    Params_this["num_feat"] = 5

    with open(Params_dir+'Params_Noise.pkl', "wb") as tf:
        pickle.dump(Params_this,tf)

def get_Params_Noise(Params_dir):
    with open(Params_dir+'Params_Noise.pkl', "rb") as tf:
        Params_this = pickle.load(tf)
    num_feat = Params_this["num_feat"]
    deg = Params_this["deg"]
    mis = Params_this["mis"]
    e_all = Params_this["e_all"]
    # alpha = Params_this["alpha"]
    num_train = Params_this["num_train"]
    return num_feat,deg,mis,e_all,num_train


def set_Params_Mis(Params_dir):
    Params_this = {}
    Params_this["deg_all"] = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,2.0,3.0,4.0]
    Params_this["e"] = 0.5
    Params_this["alpha"] = Params_this["e"]
    Params_this["num_train"] = 100
    Params_this["num_feat"] = 5
    Params_this["p"] = Params_this["num_feat"]

    with open(Params_dir+'Params_Mis.pkl', "wb") as tf:
        pickle.dump(Params_this,tf)

def get_Params_Mis(Params_dir):
    with open(Params_dir+'Params_Mis.pkl', "rb") as tf:
        Params_this = pickle.load(tf)
    num_feat = Params_this["num_feat"]
    deg_all = Params_this["deg_all"]
    # mis = Params_this["mis"]
    e = Params_this["e"]
    alpha = Params_this["alpha"]
    num_train = Params_this["num_train"]
    p = Params_this["p"]

    return num_feat,deg_all,e,num_train
