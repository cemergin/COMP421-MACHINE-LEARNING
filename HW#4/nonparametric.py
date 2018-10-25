import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#Function for getting data
def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../ML/hw04_data_set.csv')
    data = df.as_matrix()
    #Shuffling data so that learning and testing
    np.random.shuffle(data)
    #Y = data[:, 0]
    #X = data[:, 1]
    D = data[:,]
    return D

#RMSE Function
def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def RunningMeanSmootherMap(train,test, bin_width):

    W = (test - train[:,0]) / bin_width
    D = np.logical_and(W < 1.0, W > -1.0) * 1.0
    U = D * train[:,1]
    return (sum(U) / sum(D))

def KernelSmootherMap(train,test,bin_width):

    K = (test - train[:,0]) / bin_width
    D = (2*math.pi ** -0.5) * np.exp(-0.5 * np.power(K,2))
    U = D * train[:,1]

    return (sum(U) / sum(D))

def Regressogram(train, bin_width, maximum_value, minimum_value):

    #Arrange bin borders
    left_borders = np.arange(minimum_value, maximum_value - bin_width, bin_width)

    X = train[:, 0]
    Y = train[:, 1]
    R = []

    #Find average output values for each bin
    for i in xrange(len(left_borders)):
        T = np.logical_and(left_borders[i] <= X, X < left_borders[i] + bin_width)
        #Some checking done here to make sure that integers are appended to R
        if(math.isnan(sum(T)) or sum(T) == 0):
           if(len(R)>0):
                R.append(0)
           else:
               R.append(R[-1])
        else:
           R.append(sum(T*Y)/sum(T))

    R = np.array(R)
    return R, left_borders

def RegressogramMap(regresso, borders, bin_width, test):

    for i in xrange(len(borders)):
        if  test >= borders[i] and test < (borders[i] + bin_width):
            return regresso[i]
    return 0

def main():

    #Get data and split into training data and testing data
    D = get_data()
    D_train = D[:100,]
    D_test = D[100:,]

    #Get min/max value of x and y for better plotting
    minimum_value_y = min(D[:,1]) - 5
    maximum_value_y = max(D[:,1]) + 5
    minimum_value_x = min(D[:,0]) - 5
    maximum_value_x = max(D[:,0]) + 5

    #Plot initial set-up
    plt.scatter(D_train[:,0],D_train[:,1], c='b', label='Train', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', label='Test', s=50, alpha=0.8)
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #Generate x values for future plotting
    x_cont= np.arange(minimum_value_x - 1.0, maximum_value_x + 1, 0.01)

    #Test for Regressogram
    bin_width = 3
    R, left_borders = Regressogram(D_train, bin_width,60, 0)
    myRG = np.vectorize(lambda t: RegressogramMap(R,left_borders,bin_width,t))

    #Calculations Here
    y_cont = myRG(x_cont)
    prediction = myRG(D_test[:,0])

    print 'Regressogram => RMSE is ', RMSE(prediction, D_test[:,1]), ' when h is ', bin_width

    #Plot Graph
    plt.scatter(D_train[:,0],D_train[:,1], c='b', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', s=50, alpha=0.8)
    plt.plot(x_cont, y_cont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regressogram')
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.show()

    #Test for Running Mean Smoother
    bin_width = 3
    myRM = np.vectorize(lambda t: RunningMeanSmootherMap(D_train,t,bin_width))

    #Calculations Here
    y_cont = myRM(x_cont)
    prediction = myRM(D_test[:,0])

    print 'Running Mean Smoother => RMSE is ', RMSE(prediction, D_test[:,1]), ' when h is ', bin_width

    #Plot Graph
    plt.scatter(D_train[:,0],D_train[:,1], c='b', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', s=50, alpha=0.8)
    plt.plot(x_cont, y_cont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Running Mean Smoother')
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.show()

    #Test for Kernel Smoother
    bin_width = 1
    myKS = np.vectorize(lambda t: KernelSmootherMap(D_train,t,bin_width))

    #Calculations Here
    y_cont = myKS(x_cont)
    prediction = myKS(D_test[:,0])

    print 'Kernel Smoother => RMSE is ', RMSE(prediction, D_test[:,1]), ' when h is ', bin_width

    #Plot Graph
    plt.scatter(D_train[:,0],D_train[:,1], c='b', s=50, alpha=0.8)
    plt.scatter(D_test[:,0],D_test[:,1], c='g', s=50, alpha=0.8)
    plt.plot(x_cont, y_cont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kernel Smoother')
    plt.axis([minimum_value_x,maximum_value_x, minimum_value_y, maximum_value_y])
    plt.show()

if __name__ == '__main__':
    main()
