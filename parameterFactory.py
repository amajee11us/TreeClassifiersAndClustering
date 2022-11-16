from itertools import product

def fetchDTParameters():
    params = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['random', 'best'],
        'max_depth' : [None, 3, 10, 100],
        'max_features': [None, 'sqrt']
    }

    param_list = list(params.keys())

    return params, param_list

def fetchBaggingClfParameters():
    params = {
        'n_estimators': [10, 100, 1000, 2000],
        'max_samples': [10, 50, 100, 1000, 5000],
        'max_features' : [10, 100, 300, 500, 1000, 1500]
    }

    param_list = list(params.keys())

    return params, param_list

def fetchGradientBoostingClfParameters():
    params = {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.01, 0.1, 0.2, 0.5, 1.0],
        'n_estimators' : [1, 10, 100, 1000, 2000],
        'max_depth': [None, 3, 10, 100]
    }

    param_list = list(params.keys())

    return params, param_list

def fetchRandomForestClfParameters():
    params = {
        'n_estimators': [10, 100, 1000, 2000],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth' : [None, 3, 10, 100],
        'max_features': [10, 100, 300, 500, 1000, 1500]
    }

    param_list = list(params.keys())

    return params, param_list

def pickBestParams(model_name):
    '''
    This is used either to initialize the params and to store the best ones.
    The order is determined based on the order of params above.
    '''
    if model_name == "dtree":
        return ['log_loss', 'best', 10, 'None']
    elif model_name == "bagging":
        return [1000, 100, 500]
    elif model_name == "gradBoost":
        return ['log_loss', 'uniform', 1000, None]
    elif model_name == "randomForest":
        return [100, 'log_loss', 10, 500]
    else:
        return [] # return a blank array as the options dont match
