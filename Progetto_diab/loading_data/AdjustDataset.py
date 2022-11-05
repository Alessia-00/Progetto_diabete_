import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


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


def autopct(pct):
    return ('%.2f' % pct + "%") if pct > 1 else ''  # shows only values of labels that are greater than 1%


def optimizationData():
    df = pd.read_csv(r"Progetto_diab\diabetes.csv")
    pd.set_option('display.max_columns', None)
    print("Dataset loaded.")

    # OTTIMIZZAZIONE DATI

    # Eliminazione valori nulli
    print(df.describe())

    # Sostituisco gli zeri con NaN
    df.loc[df["Glucose"] == 0.0, "Glucose"] = np.NAN
    df.loc[df["BloodPressure"] == 0.0, "BloodPressure"] = np.NAN
    df.loc[df["SkinThickness"] == 0.0, "SkinThickness"] = np.NAN
    df.loc[df["Insulin"] == 0.0, "Insulin"] = np.NAN
    df.loc[df["BMI"] == 0.0, "BMI"] = np.NAN

    print(style.RED + "\nValues with zeroes." + style.RESET)
    print(df.isnull().sum()[1:6])

    print(style.YELLOW + "Filling null values..." + style.RESET)
    # Imput sui nan facendo la media
    df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
    df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
    df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace=True)
    df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
    df["BMI"].fillna(df["BMI"].mean(), inplace=True)
    print(df.isnull().sum())
    print("\n")


    print(style.YELLOW + "Check for class balance!" + style.RESET)
    """
    # vediamo quanti diab e non
    labels = ["Not diabetes", "Diabetes"]
    ax = df['Outcome'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of diabetes and non-diabetes")
    plt.legend(labels=labels, loc="best")
    plt.show()
    """
    # Proportion of non-diabetes (0) and diabetes (1):
    # [Number of non-diabetes/Total number of diabetes]
    print(style.GREEN + "Not diabetes: " + style.RESET, df.Outcome.value_counts()[0],
                      '(% {:.2f})'.format(df.Outcome.value_counts()[0] / df.Outcome.count() * 100))
    print(style.RED + "Diabetes: " + style.RESET, df.Outcome.value_counts()[1],
                    '(% {:.2f})'.format(df.Outcome.value_counts()[1] / df.Outcome.count() * 100))

    df_majority = df[df["Outcome"] == 0]
    df_minority = df[df["Outcome"] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=500, random_state=42)
    df = pd.concat([df_minority_upsampled, df_majority])

    print(style.YELLOW + "\nValue after oversampling:" + style.RESET)
    print(style.GREEN + "Not diabetes: " + style.RESET, df.Outcome.value_counts()[0],
                      '(% {:.2f})'.format(df.Outcome.value_counts()[0] / df.Outcome.count() * 100))
    print(style.RED + "Diabetes: " + style.RESET, df.Outcome.value_counts()[1],
                    '(% {:.2f})'.format(df.Outcome.value_counts()[1] / df.Outcome.count() * 100))

    """
    # Visualization of the aspect ratio chart
    labels = ["Not diabetes", "Diabetes"]
    ax = df['Outcome'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of diabetes and non-diabetes\n\nafter oversampling")
    plt.legend(labels=labels, loc="best")
    plt.show()
    
    
    # vediamo la correlazione tra le features
    # figure size
    plt.figure(figsize=(10, 10))
    # correlation matrix
    dataplot = sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.show()
    """
    return df


def loadDataset(df):

    y = df['Outcome'].values
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
            'Age']].values

    correlation_matrix = df.corr()
    correlation_matrix['Outcome'].sort_values(ascending=False)
    #print("Correlation Matrix:\n", correlation_matrix)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=25)
    return X_train, X_test, y_train, y_test, X, y, df