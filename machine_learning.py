import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
documentation of classes
use ABC
PEP8 naming convention
type casting
'''
class RegressionUtils():
    
    def normalize_z(self, df, columns_means=None, columns_stds=None):
        if columns_means is None:
            columns_means = df.mean(axis=0)

        if columns_stds is None:
            columns_stds = df.std(axis=0)
        
        dfout = (df - np.array(columns_means)) / np.array(columns_stds)
        
        return dfout, columns_means, columns_stds   
    def normalize_z(self, df, columns_means=None, columns_stds=None):
        if columns_means is None:
            columns_means = df.mean(axis=0)

        if columns_stds is None:
            columns_stds = df.std(axis=0)
        
        dfout = (df - np.array(columns_means)) / np.array(columns_stds)
        
        return dfout, columns_means, columns_stds
    def prepare_feature(self, df_feature):
        if isinstance(df_feature, pd.DataFrame):
            np_feature = df_feature.to_numpy()
            cols = df_feature.shape[1]
        elif isinstance(df_feature, pd.Series):
            np_feature = df_feature.to_numpy()
            cols = 1
        else:
            np_feature = df_feature
            cols = df_feature.shape[1]
        feature = np_feature.reshape(-1, cols)
        X = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)
        return X
    
    def prepare_target(self, df_target):
        if isinstance(df_target, pd.DataFrame):
            np_target = df_target.to_numpy()
        elif isinstance(df_target, pd.Series):
            np_target = df_target.to_numpy()
        else:
            np_target = df_target
        np_target = np_target.reshape(-1, 1)
        return np_target
    
    def get_features_targets(self, df, feature_names, target_names):
        df_feature = df.loc[:, feature_names].copy()
        df_target = df.loc[:, target_names].copy()
        return df_feature, df_target
    
    def predict_linreg(self, df_feature, beta, means=None, stds=None):
        df_feature_z = self.normalize_z(df_feature, means, stds)[0]
        X = self.prepare_feature(df_feature_z)
        pred = self.calc_linreg(X, beta)    
        return pred
    
    def calc_linreg(self, X, beta):
        return np.matmul(X, beta)
    
    def split_data(self, df_feature, df_target, random_state=None, test_size=0.5):
        
        idx_lst = df_feature.index
        #target_idx = df_target.index
        
        if random_state is not None:
            np.random.seed(random_state)
            #print(random_state)
        
        test_idx = np.random.choice(idx_lst, size=int(test_size * len(idx_lst)), replace=False)
        #target_test_idx = np.random.choice(target_idx, size=int(test_size * len(target_idx)), replace=False)
        
        train_idx = []
        
        for idx in idx_lst:
            if idx not in test_idx:
                train_idx.append(idx)
        
        df_feature_test = df_feature.loc[test_idx, :].copy()
        df_target_test = df_target.loc[test_idx, :].copy()
        df_feature_train = df_feature.loc[train_idx, :].copy()
        df_target_train = df_target.loc[train_idx, :].copy()    
        
        return df_feature_train, df_feature_test, df_target_train, df_target_test

class RegressionEvaluation():

    def r2_score(self, y, ypred):
        ssres = np.matmul((y - ypred).T, (y - ypred))
        sstot = np.matmul((y - np.mean(y)).T, (y - np.mean(y)))
        return 1 - (ssres / sstot)[0][0]
    def mean_squared_error(self, target, pred):
        m = target.shape[0]
        err = target - pred
        return np.matmul(err.T, err)[0][0] / m
    def compute_cost_linreg(self, X, y, beta):
        J = 0
        m = len(y)
        y_cap = np.matmul(X, beta) #calc_linreg(X, beta)
        J = (1 / (2 * m)) * np.matmul((y_cap - y).T, (y_cap - y))
        return J[0][0]
    def gradient_descent_linreg(self, X, y, beta, alpha, num_iters):
        m = len(y)
        J_storage = []
        for _ in range(num_iters):
            pred = calc_linreg(X, beta)
            error = pred - y
            beta = beta - alpha / m * np.matmul(X.T, error)
            J_storage.append(compute_cost_linreg(X, y, beta))
        return beta, J_storage

    

class RegressionModel(RegressionUtils, RegressionEvaluation):
    
    def __init__(self):
        self.utils = RegressionUtils()
        

    def linreg(self, df, feature_col, target_col, iterations=1500, alpha=0.01, random_state=100, test_size=0.3, sample=1):
        df_features, df_target = (df.loc[:, feature_col].copy(), df.loc[:, target_col].copy())

        df_features_train, df_features_test, df_target_train, df_target_test = self.split_data(df_features, df_target, random_state=random_state, test_size=test_size)
        df_features_train_z,_,_ = self.normalize_z(df_features_train, np.mean(df_features, axis=0), np.std(df_features, axis=0, ddof=sample))

        X = self.prepare_feature(df_features_train_z)
        target = self.prepare_target(df_target_train)

        beta = np.zeros((7,1))

        beta, J_storage = self.gradient_descent_linreg(X, target, beta, alpha, iterations)
        return beta


    def split_data_k_cross(df, random_state=None, k=10):
        
        if random_state is not None:
            np.random.seed(random_state)
            
        df_copy = df.copy().to_numpy()
        np.random.shuffle(df_copy)
        
        folds = np.array_split(df_copy, k)
        
        return folds

    def k_cross_validation(df, feature_col, target_col, k=10, iterations=1500, alpha=0.01):
        folds = split_data_k_cross(df)
        beta_lst = []
        ddfold = {}
        for i in range(k):
            ddfold[i] = pd.DataFrame(folds[i], columns=(target_col + feature_col))

        nlst = list(range(k))

        for i in range(k):
            tlst = []
            for idx in nlst:
                if idx != i:
                    tlst.append(ddfold[idx])
            test = ddfold[i]
            train = pd.concat(tlst)
            #display(test)
            #print(train.shape)
            df_features = df.loc[:, feature_col]
            df_features_train = train.loc[:, feature_col].copy()
            df_target_train = train.loc[:, target_col].copy()
            df_features_test = test.loc[:, feature_col].copy()
            df_target_test = test.loc[:, target_col].copy()

            df_features_train_z,_,_ = self.normalize_z(df_features_train, np.mean(df_features, axis=0), np.std(df_features, axis=0, ddof=1))

            X = self.prepare_feature(df_features_train_z)
            target = self.prepare_target(df_target_train)

            beta = np.zeros((7,1))

            # Call the gradient_descent function
            beta, J_storage = self.gradient_descent_linreg(X, target, beta, alpha, iterations)

            # call the predict() method
            pred = self.predict_linreg(df_features_test, beta)

            beta_lst.append(beta)
        return beta_lst

    def run_k_cross_validation(self):
            df = pd.read_csv("2D_DATA.csv")
            feature_col = ["TEMP", "TLU", "RAIN", "POP", "DEBT", "ECO"]
            target_col = ["POC"]
            beta_l = k_cross_validation(df, feature_col, target_col, k=10, iterations=1500, alpha=0.01)    
            print(beta_l)
            print(len(beta_l))
            
if __name__ == '__main__':
    RegressionModel().run_k_cross_validation()