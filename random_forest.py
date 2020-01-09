from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import numpy as np


class MyRandomForestClassifier():
    def __init__(self, n_estimators=10, max_features='sqrt'):
        self.clfs = [DecisionTreeClassifier(max_features=max_features) for _ in range(n_estimators)]
        self.labels = None

    # train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        self.labels = set(y)
        N = X.shape[0]
        for clf in self.clfs:
            indices = np.random.choice(np.arange(N), size=N, replace=True)
            clf.fit(X[indices], y[indices])


    # predict the label for each point in X
    def predict(self, X):
        predictions = [clf.predict(X) for clf in self.clfs]
        unit = {label: 0 for label in self.labels}
        votes = [unit for _ in range(X.shape[0])]

        y_pred = []
        for i in range(X.shape[0]):
            for prediction in predictions:
                vote = prediction[i]
                (votes[i])[str(vote)] += 1
            y_pred.append(max(votes[i].items(), key=lambda x: x[1])[0])

        #return np.array(y_pred)
        predictions = [tree.predict(X) for tree in self.clfs]
        return mode(predictions, axis=0)[0][0]


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    dataset = fetch_openml("mnist_784")
    X = dataset["data"]
    y = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, shuffle=True, stratify=y)

    """clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))"""

    # using a subset that is sqrt(784)
    clf = MyRandomForestClassifier(10, 28)
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))