import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def nbWithGridView(x_train, y_train):
    print(" - Execute NB with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)
    pipe = Pipeline([('sc', StandardScaler()), ('nb', GaussianNB(priors=None, var_smoothing=0.1))])
    param_grid = {'nb__var_smoothing': np.logspace(0, -9, num=100)}
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    optimal_params = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10)
    optimal_params .fit(x_train, y_train)
    return optimal_params
