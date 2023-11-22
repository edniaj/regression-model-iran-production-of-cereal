import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_z(df, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means = df.mean(axis=0)

    if columns_stds is None:
        columns_stds = df.std(axis=0)
    
    dfout = (df - np.array(columns_means)) / np.array(columns_stds)
    
    return dfout, columns_means, columns_stds

def normalize_minmax(dfin, columns_mins=None, columns_maxs=None):
#def normalize_z(df, columns_means=None, columns_stds=None):

    if columns_mins is None:
        columns_mins = dfin.min(axis=0)

    if columns_maxs is None:
        columns_maxs = dfin.max(axis=0)
    
    if isinstance(columns_maxs, list) and isinstance(columns_mins, list):
        columns_gap = []
        for idx in range(len(columns_maxs)):
            columns_gap.append(columns_maxs[idx] - columns_mins[idx])
            
        dfout = (dfin - np.array(columns_mins)) / np.array(columns_gap)
    else:
        dfout = (dfin - np.array(columns_mins)) / (np.array(columns_maxs) - np.array(columns_mins))
    
    return dfout, columns_mins, columns_maxs

def get_features_targets(df, feature_names, target_names):
    df_feature = df.loc[:, feature_names].copy()
    df_target = df.loc[:, target_names].copy()
    return df_feature, df_target

def prepare_feature(df_feature):
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

def prepare_target(df_target):
    if isinstance(df_target, pd.DataFrame):
        np_target = df_target.to_numpy()
    elif isinstance(df_target, pd.Series):
        np_target = df_target.to_numpy()
    else:
        np_target = df_target
    np_target = np_target.reshape(-1, 1)
    return np_target

def predict_linreg(df_feature, beta, means=None, stds=None):
    df_feature_z = normalize_z(df_feature, means, stds)[0]
    X = prepare_feature(df_feature_z)
    pred = calc_linreg(X, beta)
    
    return pred

def calc_linreg(X, beta):
    return np.matmul(X, beta)

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    
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
  
def r2_score(y, ypred):
    ssres = np.matmul((y - ypred).T, (y - ypred))
    sstot = np.matmul((y - np.mean(y)).T, (y - np.mean(y)))
    return 1 - (ssres / sstot)[0][0]

def mean_squared_error(target, pred):
    m = target.shape[0]
    err = target - pred
    return np.matmul(err.T, err)[0][0] / m

def compute_cost_linreg(X, y, beta):
    J = 0
    m = len(y)
    y_cap = np.matmul(X, beta) #calc_linreg(X, beta)
    J = (1 / (2 * m)) * np.matmul((y_cap - y).T, (y_cap - y))
    return J[0][0]

def gradient_descent_linreg(X, y, beta, alpha, num_iters):
    m = len(y)
    J_storage = []
    for _ in range(num_iters):
        pred = calc_linreg(X, beta)
        error = pred - y
        beta = beta - alpha / m * np.matmul(X.T, error)
        J_storage.append(compute_cost_linreg(X, y, beta))
    return beta, J_storage

df = pd.read_csv("2D_DATA.csv")
feature_col = ["TEMP", "TLU", "RAIN", "POPUL", "DEBT", "ECO"]
target_col = ["POC"]

def linreg(df, feature_col, target_col, iterations=1500, alpha=0.01, random_state=100, test_size=0.3, sample=1):
    df_features, df_target = (df.loc[:, feature_col].copy(), df.loc[:, target_col].copy())

    df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=random_state, test_size=test_size)
    df_features_train_z,_,_ = normalize_z(df_features_train, np.mean(df_features, axis=0), np.std(df_features, axis=0, ddof=sample))

    X = prepare_feature(df_features_train_z)
    target = prepare_target(df_target_train)

    beta = np.zeros((7,1))

    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
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
        df_features_train = train.loc[:, feature_col].copy()
        df_target_train = train.loc[:, target_col].copy()
        df_features_test = test.loc[:, feature_col].copy()
        df_target_test = test.loc[:, target_col].copy()

        df_features_train_z,_,_ = normalize_z(df_features_train, np.mean(df_features, axis=0), np.std(df_features, axis=0, ddof=1))

        X = prepare_feature(df_features_train_z)
        target = prepare_target(df_target_train)

        beta = np.zeros((7,1))

        # Call the gradient_descent function
        beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

        # call the predict() method
        pred = predict_linreg(df_features_test, beta)

        beta_lst.append(beta)
    return beta_lst


beta = linreg(df, feature_col, target_col, iterations=1500, alpha=0.01, random_state=100, test_size=0.3, sample=1)
pred = predict_linreg(df_features_test, beta)
k_cross_validation(df, feature_col, target_col, k=10, iterations=1500, alpha=0.01)