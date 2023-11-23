import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
Class RegressionUtils
    Provides utility functions
Class RegressionEvaluation
    Provides Calculations functions for evaluation metrics and optimzation
    
Class RegressionMode(RegressionUtils, RegressionEvaluation)
    Provides Training functions and building the CSV file 
'''
class RegressionUtils():
    
    '''
    Methods:
        normalize_z(df :pd.DataFrame, columns_means: float, columns_stds: float): /
                    -> dfout:pd.Dataframe, columns_means:float, columns_stds: float        
            Normalizes a DataFrame using Z-score normalization. It can use precomputed means and standard deviations if provided.
        
        prepare_feature(df_feature: pd.DataFrame): -> pd.DataFrame
            Prepares the feature matrix for regression by adding a constant column. It handles DataFrame, Series, and NumPy array inputs.
            
        prepare_target(df_target: pd.DataFrame): -> pd.DataFrame
            Converts target data into a suitable format for regression. Works with DataFrame, Series, and NumPy array inputs.
            
        get_features_targets(df:pd.DataFrame, feature_names:list[Str], target_names:list[Str]): 
            Extracts features and targets from a DataFrame based on specified column names.
            
        predict_linreg(df_feature: pd.DataFrame, beta: list[float], means: float, stds: float): 
            Makes predictions using a linear regression model given features, model coefficients, and optional normalization parameters.
            
        calc_linreg(X: pd.DataFrame, beta:pd.DataFrame): 
            Calculates the output of a linear regression equation given a feature matrix and model coefficients.
        
        split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, random_state: float, test_size:): 
            Splits feature and target data into training and testing sets based on a specified test size and random state.
    """
    '''
    
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
    '''
    Attributes:
        list_r2_score : List[float] 
            This is used for writing into the CSV file for k-fold training, will be used for evaluation report
        list_mean_square_error : List[float]
            This is used for writing into the CSV file for k-fold training, will be used for evaluation report
    
    Methods:    
        r2_score(y:pd.DataFrame, ypred: pd.DataFrame) -> float
            Generate te r2_sccore using the feature and target
            
        mean_squared_error(y:pd.DataFrame, ypred: pd.DataFrame) -> float
            Generate mean square error value
            
        compute_cost_linreg(X:pd.DataFrame, ypred: pd.DataFrame, beta: pd.DataFrame) -> float   
            Compute the cost of the linear regression
             
        gradient_descent_linreg(X:pd.DataFrame, y: pd.DataFrame, beta:pd.DataFrame, alpha:float, num_iters:float) -> Tuple(beta: pd.DataFarme, J_storage: list[float])
            Gradient descent linear regression algorithm
    '''
    def __init__(self):
        self.list_r2_score = []
        self.list_mean_square_error = []
        
    def r2_score(self, y, ypred):
        
        if ~isinstance(ypred, pd.DataFrame):
            ypred = pd.DataFrame(ypred)
        
        ypred.rename(columns={0:'POC'}, inplace=True)
               

        y['POC'].astype(float)
        ypred['POC'].astype(float)
        
        #residual sum of squares
        y_minus_ypred = y - ypred
        ssres = np.matmul(y_minus_ypred.T, y_minus_ypred)
        
        y_minus_mean = y - np.mean(y)
        #total sum square
        sstot = np.matmul(y_minus_mean.T, y_minus_mean)        
        ssres_divide_sstot = ssres/sstot

        r_square_value = 1 - ssres_divide_sstot.loc['POC','POC']
        self.list_r2_score.append(r_square_value)

        return r_square_value
    
    def mean_squared_error(self, y, ypred):
    
        # edit
        if ~isinstance(ypred, pd.DataFrame):
            ypred = pd.DataFrame(ypred)
        
        ypred.rename(columns={0:'POC'}, inplace=True)
               
        
        
        y['POC'].astype(float)
        ypred['POC'].astype(float)
        #
        m = y.shape[0]

        err = y - ypred
        mse_value = np.matmul(err.T, err).loc['POC','POC'] #divide by m
        mse_value /= m
        
        self.list_mean_square_error.append(mse_value)
        return mse_value
    
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
            pred = self.calc_linreg(X, beta)
            error = pred - y
            beta = beta - alpha / m * np.matmul(X.T, error)
            J_storage.append(self.compute_cost_linreg(X, y, beta))
        return beta, J_storage

class RegressionModel(RegressionUtils, RegressionEvaluation):
    '''
        Context: 
            Provides Training functions and building the CSV file 
        
        Attributes: 
            df_kfold: pd.DataFrame
                Stores the data collected from the k-fold algorithm
        
        Methods:
            write_beta_list_to_csv(array_beta: List[float]) ->
                Clean up the beta values before building a dataframe, generate CSV for models by the k-fold algo
            
            linreg(df :pd.DataFrame, feature_col: list[Str], target_col: list[Str], k:int, iterations:int, alpha:float) -> float
                Linear regression using data that will be split into training and test
            
            split_data_k_cross(df:pd.DataFrame, random_state:float, k:int) -> np.array
                Split the DataFrame into k folds for cross-validation, returns A list of DataFrames, each representing a fold.
            
            k_cross_validation(df:pd.DataFrame, feature_col: list[Str], target_col: list[Str], k: int, iterations:int, alpha: float) -> List[List[List[int]]]
                Perform k-fold cross-validation on the given DataFram and return list of model coefficients (beta values) for each fold
            
            run_k_cross_validation
                Run k-fold cross-validation on the dataset specified in the "2D_DATA.csv" file.
            
            build_csv_with_kfold
                Build a CSV file with the results from k-fold cross-validation.
                This method runs k-fold cross-validation and writes the results, including beta coefficients,
                R-squared values, and mean squared error, to a CSV file named 'k_fold.csv'
            
    '''
    def __init__(self):
        super().__init__()        
        df_kfold_columns = ["CONSTANT","TEMP", "TLU", "RAIN", "POP", "DEBT", "ECO","MSE","RSQUARE"]
        self.df_kfold = pd.DataFrame(df_kfold_columns)
        
    def write_beta_list_to_csv(self, array_beta):
        
        '''
        clean up multi-dimensional array
        array_beta = list[list[list[int]]]        
        '''
        new_array_beta = []
        for i in array_beta:
            flatten_list = []
            for j in i:
                flatten_list.append(j[0])
            new_array_beta.append(flatten_list)
            
        #["TEMP", "TLU", "RAIN", "POP", "DEBT", "ECO"]
        dict_to_dataframe = {
            'CONSTANT' : [],
            'POP' : [],
            'TEMP' : [],
            'FDI': [],
            'R2_SCORE': [],
            'MSE_SCORE': []
        }
        
        for index,each_beta_list in enumerate(new_array_beta):
            dict_to_dataframe['CONSTANT'].append(each_beta_list[0])
            dict_to_dataframe['POP'].append(each_beta_list[1])
            dict_to_dataframe['TEMP'].append(each_beta_list[2])
            dict_to_dataframe['FDI'].append(each_beta_list[3])
            dict_to_dataframe['R2_SCORE'].append(self.list_r2_score[index])
            dict_to_dataframe['MSE_SCORE'].append(self.list_mean_square_error[index])

        df_k_fold = pd.DataFrame(dict_to_dataframe)        
        print(df_k_fold)
        df_k_fold.to_csv('k_fold.csv', index=False) 
            
    def linreg(self, df, feature_col, target_col, m, iterations=1500, alpha=0.01, random_state=100, test_size=0.2, sample=1):
        df_features, df_target = (df.loc[:, feature_col].copy(), df.loc[:, target_col].copy())

        df_features_train, df_features_test, df_target_train, df_target_test = self.split_data(df_features, df_target, random_state=random_state, test_size=test_size)
        df_features_train_z,_,_ = self.normalize_z(df_features_train, np.mean(df_features, axis=0), np.std(df_features, axis=0, ddof=sample))

        X = self.prepare_feature(df_features_train_z)
        target = self.prepare_target(df_target_train)

        beta = np.zeros((m+1,1))

        beta, J_storage = self.gradient_descent_linreg(X, target, beta, alpha, iterations)
        pred = self.predict_linreg(df_features_test, beta)

        return beta, df_features_test, df_target_test, pred

    def split_data_k_cross(self, df, random_state=None, k=5):
        
        if random_state is not None:
            np.random.seed(random_state)
            
        df_copy = df.copy().to_numpy()
        np.random.shuffle(df_copy)
        
        folds = np.array_split(df_copy, k)
        
        return folds

    def k_cross_validation(self,df, feature_col, target_col,m, k=10, iterations=1500, alpha=0.01, rs=100):
        folds = self.split_data_k_cross(df, k=k, random_state=rs) #(data, number of splits, random_state)
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

            beta = np.zeros((m+1,1))

            # Call the gradient_descent function
            beta, J_storage = self.gradient_descent_linreg(X, target, beta, alpha, iterations)

            # call the predict() method
            pred = self.predict_linreg(df_features_test, beta)
            
            # Using metrics for model evaluation
            self.r2_score(df_target_test, pred)
            self.mean_squared_error(df_target_test, pred)
            # Let's store the values inside a dataframe so that we can output into a csv
            

            beta_lst.append(beta)
        return beta_lst

    def run_k_cross_validation(self, rs=100):
        
        df = pd.read_csv("variation_2_2D_DATA.csv")
        feature_col = ["POP", "TEMP", "FDI"]
        target_col = ["POC"]
        beta_l = self.k_cross_validation(df, feature_col, target_col,m=3, k=5, iterations=10000, alpha=0.01, rs=rs)
        return beta_l
        
    def build_csv_with_kfold(self, rs=100):
        
        beta_l = self.run_k_cross_validation(rs)             
        self.write_beta_list_to_csv(beta_l)

    def run_linreg_with_plot(self):
        df = pd.read_csv("variation_2_2D_DATA.csv")
        feature_col = ["POP", "TEMP", "FDI"]
        target_col = ["POC"]
        beta, test, target, pred = self.linreg(df, feature_col, target_col, m=3, iterations=1500, alpha=0.01, random_state=100, test_size=0.2, sample=1)
        
        for name in feature_col:
            plt.scatter(test[name], target)
            plt.scatter(test[name], pred)
            plt.show()
                
            
if __name__ == '__main__':
    '''
    We will use this module for out bonus page.
    Only run this function once for training the model. Do not run this if it's used as a module .
    '''
    RegressionModel().build_csv_with_kfold(rs=None)
    RegressionModel().run_linreg_with_plot()