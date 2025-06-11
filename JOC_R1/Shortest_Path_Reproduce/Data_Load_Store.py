import pickle

class Load_Store_Methods:
    def __init__(self):
        pass

    def store_input_data(self,DataPath,x_test_all,c_test_all,x_train_all,c_train_all,noise_test_all,noise_train_all,W_star_all):
            with open(DataPath+'x_test_all.pkl', "wb") as tf:
                pickle.dump(x_test_all,tf)
            with open(DataPath+'c_test_all.pkl', "wb") as tf:
                pickle.dump(c_test_all,tf)
            with open(DataPath+'x_train_all.pkl', "wb") as tf:
                pickle.dump(x_train_all,tf)
            with open(DataPath+'c_train_all.pkl', "wb") as tf:
                pickle.dump(c_train_all,tf)
            with open(DataPath+'noise_train_all.pkl', "wb") as tf:
                pickle.dump(noise_train_all,tf)
            with open(DataPath+'noise_test_all.pkl', "wb") as tf:
                pickle.dump(noise_test_all,tf)
            with open(DataPath+'W_star_all.pkl', "wb") as tf:
                pickle.dump(W_star_all,tf)

    def load_cost_data(self,DataPath):
        with open(DataPath+'cost_OLS_Post_all.pkl', "rb") as tf:
            cost_OLS_Post_all = pickle.load(tf)
        with open(DataPath+'cost_OLS_Ante_all.pkl', "rb") as tf:
            cost_OLS_Ante_all = pickle.load(tf)

        with open(DataPath+'cost_Oracle_Post_all.pkl', "rb") as tf:
            cost_Oracle_Post_all = pickle.load(tf)
        with open(DataPath+'cost_Oracle_Ante_all.pkl', "rb") as tf:
            cost_Oracle_Ante_all = pickle.load(tf)

        with open(DataPath+'cost_DDR_Post_all.pkl', "rb") as tf:
            cost_DDR_Post_all = pickle.load(tf)
        with open(DataPath+'cost_DDR_Ante_all.pkl', "rb") as tf:
            cost_DDR_Ante_all = pickle.load(tf)

        return cost_Oracle_Post_all,cost_Oracle_Ante_all,cost_OLS_Post_all,cost_OLS_Ante_all,cost_DDR_Post_all,cost_DDR_Ante_all
    

    def load_input_data(self,DataPath):
        with open(DataPath+'x_test_all.pkl', "rb") as tf:
            x_test_all = pickle.load(tf)
        with open(DataPath+'c_test_all.pkl', "rb") as tf:
            c_test_all = pickle.load(tf)
        with open(DataPath+'x_train_all.pkl', "rb") as tf:
            x_train_all = pickle.load(tf)
        with open(DataPath+'c_train_all.pkl', "rb") as tf:
            c_train_all = pickle.load(tf)
        with open(DataPath+'noise_train_all.pkl', "rb") as tf:
            noise_train_all = pickle.load(tf)
        with open(DataPath+'noise_test_all.pkl', "rb") as tf:
            noise_test_all = pickle.load(tf)
        with open(DataPath+'W_star_all.pkl', "rb") as tf:
            W_star_all = pickle.load(tf)

        return x_test_all,c_test_all,x_train_all,c_train_all,noise_train_all,noise_test_all,W_star_all
    
    def store_Oracle_OLS_DDR_Cost(self,DataPath,cost_Oracle_Post_all,cost_Oracle_Ante_all,\
                                  cost_OLS_Post_all,cost_OLS_Ante_all,\
                                  cost_DDR_Post_all,cost_DDR_Ante_all):
        with open(DataPath+'cost_Oracle_Post_all.pkl', "wb") as tf:
            pickle.dump(cost_Oracle_Post_all,tf)
        with open(DataPath+'cost_Oracle_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_Oracle_Ante_all,tf)

        with open(DataPath+'cost_OLS_Post_all.pkl', "wb") as tf:
            pickle.dump(cost_OLS_Post_all,tf)
        with open(DataPath+'cost_OLS_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_OLS_Ante_all,tf)

        with open(DataPath+'cost_DDR_Post_all.pkl', "wb") as tf:
            pickle.dump(cost_DDR_Post_all,tf)
        with open(DataPath+'cost_DDR_Ante_all.pkl', "wb") as tf:
            pickle.dump(cost_DDR_Ante_all,tf)