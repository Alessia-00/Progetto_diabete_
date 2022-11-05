from sklearn.linear_model import LogisticRegression
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


def log_regWithGridView(x_train, y_train):
    print(" - Execute LR with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)

    pipe = Pipeline([('sc', StandardScaler()), ('logr', LogisticRegression(C=0.1, penalty='l2'))])
    # Create the param grid
    param_grid = {'logr__penalty': ['l2'],
                  'logr__C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100, 1000]}

    optimal_params = GridSearchCV(estimator=pipe,
                                  param_grid=param_grid,
                                  cv=5,  # we are taking 5-fold as in k-fold cross validation
                                  scoring='accuracy')

    optimal_params.fit(x_train, y_train)
    return optimal_params
