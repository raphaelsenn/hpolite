# hpolite
You are interested in hyperparameter optimization? You like micrograd? Then You love hpolite!

A tiny (educational) Hyperparameter optimization libary that builts on top of sklearn.

hpolite implements the following search algorithms:

* GridSearchCV
* RandomSearchCV
* BayesianSearchCV (Bayesian optimization)
* HalvingRandomSearchCV  (Multi-fidelity algorithm)
* HyperbandSearchCV  (Multi-fidelity algorithm)
* EvolutionSearchCV (Genetic algorithm)

hpolite implements the following surrogate models:

* GaussianProcess
* RandomForest (TODO)
* Bayesian Neural Network (TODO)

hpolite implements the following aquisition functions:

* Probability of improvement
* Expected improvement
* Lower confidence bound
* Thompson Sampling (TODO)
* Entropy search (TODO)
* Knowledge Gradient (TODO)

## Usage

```python
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hpolite import BayesianSearchCV, Categorical, Real
from hpolite.aquisition import EI, # or PI, ES, KG, TS 
from hpolite.surrogates import GaussianProcess # or RandomForest

X, y = load_breast_cancer(return_X_y=True)  
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    train_size=0.7,
    random_state=0
)

# Define the search space
param_grid = {
    'C':        Real(1e0, 1e3, prior="log-uniform"), 
    'gamma':    Real(1e-5, 1e0, prior="log-uniform"), 
    'kernel':   Categorical(['rbf'])
}  

surrogate = GaussianProcess()
aquisition_func = EI(surrogate)
bs = BayesianSearchCV(SVC(), surrogate, aquisition_func, param_grid)
bs.fit(X_train, y_train)

print(bs.best_params_)
print(bs.best_score_)
```

## Citations

```bibtex
@book{automl,
    editor = {Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
    publisher = {Springer},
    title = {Automatic Machine Learning: Methods, Systems, Challenges},
    year = {2019}
}

@inproceedings{feurer_hyperparameter_2019,
    author = {Feurer, Matthias and Hutter, Frank},
    title = {Hyperparameter Optimization},
    pages = {3-38},
    chapter = {1},
    crossref = {automl}
}

@inproceedings{vanschoren_meta_2019,
    author = {Vanschoren, Joaquin},
    title = {Meta-Learning},
    pages = {39-68},
    chapter = {2},
    crossref = {automl}
}
```