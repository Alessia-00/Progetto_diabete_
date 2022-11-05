import warnings
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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


def mlpWithGridView(x_train, y_train):
    print(" - Execute MPL with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)
    pipe = Pipeline([('sc', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(17,), solver='lbfgs', max_iter=500))])
    #param_grid = {'mlp__activation': ['relu', 'logistic', 'tanh', 'identity']}
    param_grid = [
        {
            'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp__solver': ['lbfgs', 'sgd', 'adam'],
            'mlp__hidden_layer_sizes': [
                (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
                (17,), (18,), (19,), (20,), (21,)
            ]
        }
    ]
    #search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, refit=True, error_score=0, n_jobs=-1)
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, scoring='accuracy')
    search.fit(x_train, y_train)
    return search
