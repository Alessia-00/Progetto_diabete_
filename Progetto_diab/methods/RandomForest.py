import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def rfcWithGridView(x_train, y_train):
    print(" - Execute RFC with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)
    pipe = Pipeline([('sc', StandardScaler()), ('rfc', RandomForestClassifier())])
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]

    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']

    # Maximum number of levels in tree
    max_depth = range(1, 10)

    # measure the quality of a split
    criterion = ['gini', 'entropy']

    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the param grid

    param_grid = {'rfc__n_estimators': n_estimators,
                  'rfc__max_features': max_features,
                  'rfc__max_depth': max_depth,
                  'rfc__criterion': criterion,
                  'rfc__bootstrap': bootstrap}

    optimal_params = GridSearchCV(estimator=pipe,
                                  param_grid=param_grid,
                                  cv=5,  # we are taking 5-fold as in k-fold cross validation
                                  scoring='accuracy')
    optimal_params.fit(x_train, y_train)
    return optimal_params
