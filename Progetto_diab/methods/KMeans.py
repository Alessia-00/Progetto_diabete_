from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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


def kmns(df, infoPrint):

    # K-MEANS CLASSIFIER
    if infoPrint:
        print("\n-- ", style.BLUE + "K-MEANS" + style.RESET)

    new_df = df.drop("Outcome", axis=1)
    new_df = new_df.drop(["Pregnancies", "BloodPressure", "SkinThickness"], axis=1)
    scaler = StandardScaler()
    new_df[['Glucose_T', 'Insulin_T', 'BMI_T', 'DiabetesPedigreeFunction_T', 'Age_T']] = scaler.fit_transform(new_df)
    print(style.YELLOW + "New columns in dataset!" + style.RESET)
    print(new_df.head())
    print("\n")

    #create function to work out optimum number of clusters
    def optimise_k_means(data, max_k):
        means = []
        inertias = []

        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)

            means.append(k)
            inertias.append(kmeans.inertia_)

        #generate the elbow plot
        fig = plt.subplots(figsize=(10, 5))
        plt.plot(means, inertias, 'o-')
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

    print("Number of cluster:", style.GREEN + "3" + style.RESET)
    optimise_k_means(new_df[['Glucose_T', 'BMI_T']], 10)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit_predict(new_df[['BMI_T', 'Glucose_T']])
    new_df['labels'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(style.YELLOW + "Cluster complete!" + style.RESET)
    print(new_df.head(20))

    #print the results
    plt.scatter(x=new_df['BMI'], y=new_df['Glucose'], c=new_df['labels'])
    plt.show()

    return kmeans
