import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Function for getting data
def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../ML/hw05_data_set.csv')
    data = df.as_matrix()
    #Shuffling data so that learning and testing
    np.random.shuffle(data)
    #Y = data[:, 0]
    #X = data[:, 1]
    D = data[:,]
    return D

class TreeNode:
    def __init__(self, pruning):
        self.pruning = pruning

    def fit(self, X):

        #leaf node data size is smaller than pre-pruning value
        if X.size <= self.pruning:
            self.split = None
            self.left = None
            self.right = None
            self.prediction = np.mean(X[:,1])

        else:

            #search best x value to split
            min_variance = float("inf")
            best_split = None
            for i in xrange(len(X[:,0])):

                var = self.find_split(X,X[i,0])

                #better value found change min_variance and best_split value
                if var < min_variance:
                    min_variance = var
                    best_split = X[i,0]

            #check if splitting further will increase accuracy by checking if min_variance is changed
            if min_variance == float("inf"):
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.mean(X[:,1])

            #if splitting required we split and create left & right nodes recursively
            else:
                    self.split = best_split

                    leftIndex = np.where(X[:,0] < best_split)
                    Xleft = X[leftIndex]
                    self.left = TreeNode(self.pruning)
                    self.left.fit(Xleft)

                    rightIndex = np.where(X[:,0] >= best_split)
                    Xright = X[rightIndex]
                    self.right = TreeNode(self.pruning)
                    self.right.fit(Xright)

    #helper function which calculates E'm
    def find_split(self, X, i):

        length = len(X[:,0])

        #split X
        leftIndex = np.where(X[:,0] < i)
        Xleft = X[leftIndex]

        rightIndex = np.where(X[:,0] >= i)
        Xright = X[rightIndex]

        #Check if such split makes sense
        if len(Xleft) == length or len(Xright) == length or length == 0:
            return float("inf")

        #calculate variances
        varLeft = np.var(Xleft[:,1])
        varRight = np.var(Xright[:,1])

        return (varLeft + varRight) / length

    def predict_one(self, x):
        if self.split is not None:
            if x < self.split:
                if self.left:
                    p = self.left.predict_one(x)
            else:
                if self.right:
                    p = self.right.predict_one(x)
        else:
            p = self.prediction
        return p

    def predict(self, X):
        length = len(X)
        P = np.zeros(length)
        for i in xrange(length):
            P[i] = self.predict_one(X[i])
        return P

class DecisionTree:
    def __init__(self, pruning):
        self.pruning = pruning

    def fit(self, X):
        self.root = TreeNode(pruning=self.pruning)
        self.root.fit(X)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return RMSE(P,Y)

#RMSE Function
def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Main function
def main():

    D = get_data()
    D_train = D[:100,]
    D_test = D[100:,]

    #Get min/max value of x and y for better plotting
    minimum_value_y = min(D[:,1]) - 5
    maximum_value_y = max(D[:,1]) + 5
    minimum_value_x = min(D[:,0]) - 1
    maximum_value_x = max(D[:,0]) + 1

    #Plot initial set-up
    plt.scatter(D_train[:,0],D_train[:,1], c='b', label='Train', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', label='Test', s=50, alpha=0.8)
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #Generate x values for future plotting
    x_cont= np.arange(minimum_value_x - 1.0, maximum_value_x + 1, 0.01)

    #train & predict
    P = 10
    model = DecisionTree(pruning=P)
    model.fit(D_train)
    y_cont = model.predict(x_cont)

    print 'RMSE is ', model.score(D_test[:,0],D_test[:,1]), ' when P is ', P

    #Plot Graph
    plt.scatter(D_train[:,0],D_train[:,1], c='b', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', s=50, alpha=0.8)
    plt.plot(x_cont, y_cont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Regression Tree')
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.show()

    x_test= np.arange(1, 31, 1)
    rmse_test = np.zeros(len(x_test))

    for i in range(x_test.size):
        model = DecisionTree(pruning=x_test[i])
        model.fit(D_train)
        score = model.score(D_test[:,0],D_test[:,1])
        #print 'RMSE is ', score, ' when P is ', x_test[i]
        rmse_test[i] = score

    plt.scatter(x_test, rmse_test, c='r', s=50, alpha=0.8)
    plt.plot(x_test, rmse_test)
    plt.xlabel('P')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. P')
    plt.show()

if __name__ == '__main__':
    main()
