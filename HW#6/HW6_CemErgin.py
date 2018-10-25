import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sortedcontainers import SortedList
from sklearn.preprocessing import normalize
import sys

def get_training_data(limit=None):
    print("Reading in and transforming training data...")
    df = pd.read_csv('../ML/hw06_mnist_training_digits.csv',header=None)
    data = df.as_matrix()
    D = data[:,]

    df2 = pd.read_csv('../ML/hw06_mnist_training_labels.csv', header=None)
    data2 = df2.as_matrix()
    L = data2[:,]

    return D, L

def get_testing_data(limit=None):
    print("Reading in and transforming testing data...")
    df = pd.read_csv('../ML/hw06_mnist_test_digits.csv', header = None)
    data = df.as_matrix()
    D = data[:,]

    df = pd.read_csv('../ML/hw06_mnist_test_labels.csv', header = None)
    data = df.as_matrix()
    L = data[:,]

    return D, L

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))
    return X, Y

class KNN(object):

    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i in xrange(len(X)):
            x = X[i]
            sl = SortedList()
            for j in xrange(len(self.X)):
                xt = (self.X)[j]
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))

            votes = {}
            for _, v in sl:
                if type(v).__name__ == 'int':
                    k = v
                else:
                    k = v.item(0)
                if (votes.has_key(k)):
                    votes[k] = votes[k] + 1
                else:
                    votes[k] = 1

            max_votes = -1
            max_votes_class = -1
            for v,count in votes.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v

            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

class myLDA:

    def __init__(self):

        self.eigen_list = None

    def fit(self, D, L):

        classes = np.unique(L)
        _, dimension = D.shape

        means = {}
        for label in classes:
            index = np.where(L == label)[0]
            DM = D[index]
            means[label] = np.mean(DM, axis = 0)

        overall_mean = np.mean(D, axis = 0)

        #print overall_mean

        Sb = np.zeros((dimension,dimension))

        for label in classes:
            n = len(np.where(L == label)[0])
            mv = means[label].reshape(dimension,1)
            ovm = overall_mean.reshape(dimension,1)
            Sb += n * (mv - ovm).dot((mv-ovm).T)

        Sw = np.zeros((dimension,dimension))

        for label in classes:
            index = np.where(L == label)[0]
            class_scatter = np.zeros((dimension,dimension))
            mv = means[label].reshape(dimension, 1)
            for row in index:
                mrow =  D[row].reshape(dimension, 1)
                class_scatter += (mrow - mv).dot((mrow - mv).T)
            Sw += class_scatter

        a = np.zeros((dimension,dimension))
        np.fill_diagonal(a,10^-100)
        Sw += a

        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

        self.eigen_list = eigen_pairs

    def calculate_w(self, dimension):

        self.w = np.array([self.eigen_list[i][1] for i in range(dimension)]).real
        return

        #W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
        #np.array([self.eigen_list[i][1] for i in range(dimension)]).real

    def reduce(self,D):
        return D.dot(np.transpose(self.w))

    def return_w(self):
        return self.w


def main():

    colors = ['#FF6347', '#4682B4', '#00FF7F', '#708090', '#FAA460', '#800080', '#FFC0CB', '#AFEEEE', '#FFE4E1', '#C71585', '#800000', '#FF00FF', '#F0E68C', '#FF69B4']

    D_donut, L_donut = get_donut()

    plt.scatter(D_donut[:,0], D_donut[:,1], s=100, c=L_donut, alpha=0.5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Donut Case')
    plt.show()

    model = KNN(5)
    model.fit(D_donut, L_donut)
    print "Accuracy:", model.score(D_donut, L_donut)

    D_train, L_train = get_training_data()
    D_test, L_test = get_testing_data()

    ananas = myLDA()
    ananas.fit(D_train, L_train)
    ananas.calculate_w(2)

    classes = np.unique(L_train)

    D_new = ananas.reduce(D_train)

    for label in classes:
        index = np.where(L_train == label)[0]
        D_draw = D_new[index]
        plt.scatter(D_draw[:,0], D_draw[:,1], c=colors[label], s=50, alpha=0.8)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Training Points')
    plt.show()

    D_new = ananas.reduce(D_test)

    for label in classes:
        index = np.where(L_train == label)[0]
        D_draw = D_new[index]
        plt.scatter(D_draw[:,0], D_draw[:,1], c=colors[label], s=50, alpha=0.8)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Test Points')
    plt.show()

    ananas = myLDA()
    ananas.fit(D_train, L_train)
    avacado = KNN(5)
    test = range(0,10)
    score = []
    ind = []

    for i in test:
        ind.append(i)
        ananas.calculate_w(i+1)
        red = ananas.reduce(D_train)
        avacado.fit(red,L_train)
        der = ananas.reduce(D_test)
        score.append(avacado.score(der,L_test))

    plt.plot(ind, score,'o')
    plt.xlabel('R')
    plt.ylabel('Classification Accuracy (%)')
    plt.axis([-1, 12, 0, 1])
    plt.show()

if __name__ == '__main__':
    main()
