from sklearn.neighbors import KNeighborsClassifier
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


def kncWithGridView(x_train, y_train):

    print(" - Execute KNC with Grid View")
    print(style.YELLOW + "\tCalculating optimal hyperparameters..." + style.RESET)
    pipe = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(algorithm='kd_tree', n_neighbors=120, p=2, weights='distance'))])
    param_grid = {'knn__n_neighbors': [5, 10, 15, 30, 60, 90, 120],
                  'knn__weights': ['uniform', 'distance'],
                  'knn__algorithm': ['kd_tree', 'ball_tree', 'brute'],
                  'knn__p': [1, 2]}
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='accuracy', return_train_score=False)
    search.fit(x_train, y_train)
    return search
