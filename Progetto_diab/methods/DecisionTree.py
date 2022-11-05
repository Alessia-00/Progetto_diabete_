from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# Class of different styles
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


def dtcWithGridView(x_train, y_train):
    print(" - Execute DCT with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)
    pipe = Pipeline([('scaler', StandardScaler()), ('dtc', DecisionTreeClassifier(criterion='gini', max_depth=15))])
    param_grid = {'dtc__criterion': ['gini', 'entropy'],
                  'dtc__max_depth': range(1, 100)}
    optimal_params = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='accuracy')
    optimal_params.fit(x_train, y_train)
    return optimal_params
