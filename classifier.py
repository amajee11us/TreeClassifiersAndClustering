import os
from symbol import parameters
import numpy as np
import math

# sklearn libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from parameterFactory import fetchDTParameters, fetchBaggingClfParameters, fetchGradientBoostingClfParameters, fetchRandomForestClfParameters, pickBestParams
from utils import getPerformanceScores

# grid search
from sklearn.model_selection import RandomizedSearchCV

# display results
from prettytable import PrettyTable

class TreeClassifier:
    def __init__(self, model_name='dtree'):
        self.model_name = model_name
        '''
        Fetch the parameter iterator based on the model family
        Note : Each iterator returns a different list of parameters 
        '''
        if self.model_name == 'dtree':
            self.iterator, self.param_names = fetchDTParameters()
        elif self.model_name == 'bagging':
            self.iterator, self.param_names = fetchBaggingClfParameters()
        elif self.model_name == "gradBoost":
            self.iterator, self.param_names = fetchGradientBoostingClfParameters()
        elif self.model_name == "randomForest":
            self.iterator, self.param_names = fetchRandomForestClfParameters()
        else:
            print("[Error] Your choice of classifier does not exist.")
        
        self.pred_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        #column names to display
        self.table_columns = [x for x in self.param_names]
        self.table_columns.extend(self.pred_metrics)

    def __init_model(self, parameters):
        '''
        Generates a model based on two paramters:
        model_name : dtree/ bagging/ gradientboosting/ randomForest 
        parameters : choice of parameters passed to the model from iterator
        returns:
        An object of the model with the parameters baked into it
        '''
        if self.model_name == 'dtree':
            '''
            Parameters (in-order) : criterion, splitter, max_depth, max_features, ccp_alpha
            '''
            self.classifier = DecisionTreeClassifier(criterion=parameters[0],
                                                     splitter=parameters[1],
                                                     max_depth=parameters[2],
                                                     max_features=parameters[3]
                                                    )
        elif self.model_name == 'bagging':
            self.classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                n_estimators=parameters[0], 
                                                max_samples=parameters[1],
                                                max_features=parameters[2]
                                                )
        elif self.model_name == "gradBoost":
            self.classifier = GradientBoostingClassifier(loss=parameters[0], 
                                                         learning_rate=parameters[1], 
                                                         n_estimators=parameters[2],
                                                         max_depth=parameters[3],
                                                         tol=1e-7
                                                        )
        elif self.model_name == "randomForest":
            self.classifier = RandomForestClassifier(n_estimators=parameters[0], 
                                                     criterion=parameters[1], 
                                                     max_depth=parameters[2],
                                                     max_features=parameters[3],
                                                     verbose=1
                                                    )
        else:
            print("[Error] Your choice of classifier does not exist.")

    def tune(self, X_train, Y_train, X_test, Y_test):
        '''
        Performs model tuning
        1. Choose set of parameters from iterator present in parameterFactory.py
        2. Initialize a new model with the parameters in step 1
        3. Run training and validation on the model
        4. Display a table of results from all possible model combination
        '''
        tab = PrettyTable(self.table_columns)
        best_params = [] # store the best performing model based on accuracy
        best_acc = 0.0
        exp_count = 1

        '''
        Initialize the classifier - The first value in the parameter list
        '''
        initial_params = pickBestParams(self.model_name)
        self.__init_model(initial_params)
        '''
        Initialize the Parameter Search - Greedy but randomized search
        '''
        print("---------------------------------------")
        print("Running Model Type : {}".format(self.model_name))
        param_searcher = RandomizedSearchCV(self.classifier, self.iterator, 
                                            n_iter = 10, random_state = 0, verbose = 10)
        '''
        Search for the best parameters using a Randomized Greedy Search
        '''
        search = param_searcher.fit(X_train, Y_train)
        
        print("Final Results of Ablation experiments ...")
        best_params = [search.best_params_[param_name] for param_name in self.param_names]
        predictions = search.predict(X_test)
        acc, pr, recall, f1 = getPerformanceScores(predictions, Y_test)
        best_params.extend(np.round([acc, pr, recall, f1],3))
        tab.add_row(best_params)
        print(tab)
        print("Done.")

        return best_params

    def trainval(self, X_train, Y_train, X_test, Y_test, params):
        '''
        Train and test a classifier with only a single parameter set
        Note: This is only called after the tuning process
        '''
        # Initialize the model with the passed params
        self.__init_model(params)
        # Train the model on the complete val set
        self.classifier.fit(X_train, Y_train)
        # Return the evaluated results on the complete test set
        return self.classifier.predict(X_test)
