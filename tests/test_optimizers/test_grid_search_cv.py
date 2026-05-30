import pytest

import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV as GridSearchCV

from hpolite.optimizers.grid_search_cv import GridSearchCV as GridSearchCVLite
from hpolite.space.categorical import Categorical


class TestGridSearchCV:
    train_size = 0.7
    random_state = 0
    cv = 5
    def test_breast_cancer_svm(self) -> None:
        X, y = load_breast_cancer(return_X_y=True)  

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            train_size=self.train_size, 
            random_state=self.random_state
        )
        
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
			'kernel': ['rbf']
        }  

        param_grid_hpolite = {
            'C': Categorical([0.1, 1, 10, 100, 1000]),
			'gamma': Categorical([1, 0.1, 0.01, 0.001, 0.0001]),
			'kernel': Categorical(['rbf'])
        }  
        grid_sklearn = GridSearchCV(SVC(), param_grid, cv=self.cv, refit = True) 
        grid_sklearn.fit(X_train, y_train)

        grid_hpolite = GridSearchCVLite(SVC(), param_grid_hpolite, cv=self.cv)
        grid_hpolite.fit(X_train, y_train)

        assert grid_sklearn.best_params_ == grid_hpolite.best_params_
        assert grid_sklearn.best_score_ == grid_hpolite.best_score_