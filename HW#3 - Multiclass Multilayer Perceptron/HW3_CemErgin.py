import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(421)

#FUNCTIONS

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z

# Will help determine accuracy of predictions
# correct / total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    ret1 = Z.T.dot(T - Y)

    return ret1

def derivative_b2(T,Y):
    return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)

    return ret2

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def cost(T,Y):
    tot = T * np.log(Y)
    return tot.sum()

#MAIN

def main():

    Nclass = 100
    D = 2 #number of dimensions
    M = 10 #size of hidden layer
    K = 4 #number of classes

    #Generate Points for group 1
    cov1 = np.array([[0.8,-0.6],[-0.6,0.8]])
    mu1 = np.array([2.0,2.0])
    x1 = np.random.multivariate_normal(mean = mu1, cov=cov1,size=50)

    cov2 = np.array([[0.4,0.0],[0.0,0.4]])
    mu2 = np.array([-4.0,-4.0])
    x2 = np.random.multivariate_normal(mean = mu2, cov=cov2,size=50)
    g1 = np.vstack([x1,x2])

    #Generate Points for group 2
    cov3 = np.array([[0.8,0.6],[0.6,0.8]])
    mu3 = np.array([-2.0,2.0])
    x3 = np.random.multivariate_normal(mean = mu3, cov=cov3,size=50)

    cov4 = np.array([[0.4,0.0],[0.0,0.4]])
    mu4 = np.array([4.0,-4.0])
    x4 = np.random.multivariate_normal(mean = mu4, cov=cov4,size=50)
    g2 = np.vstack([x3,x4])

    #Generate Points for group 3
    cov5 = np.array([[0.8,-0.6],[-0.6,0.8]])
    mu5 = np.array([-2.0,-2.0])
    x5 = np.random.multivariate_normal(mean = mu5, cov=cov5,size=50)

    cov6 = np.array([[0.4,0.0],[0.0,0.4]])
    mu6 = np.array([4.0,4.0])
    x6 = np.random.multivariate_normal(mean = mu6, cov=cov6,size=50)
    g3 = np.vstack([x5,x6])

    #Generate Points for group 4
    cov7 = np.array([[0.8,0.6],[0.6,0.8]])
    mu7 = np.array([2,-2])
    x7 = np.random.multivariate_normal(mean = mu7, cov=cov7,size=50)

    cov8 = np.array([[0.4,0],[0,0.4]])
    mu8 = np.array([-4,4])
    x8 = np.random.multivariate_normal(mean = mu8, cov=cov8,size=50)
    g4 = np.vstack([x7,x8])

    '''
    #Plot all samples with different color
    plt.scatter(g1[:,0],g1[:,1],color='r')
    plt.scatter(g2[:,0],g2[:,1],color='g')
    plt.scatter(g3[:,0],g3[:,1],color='b')
    plt.scatter(g4[:,0],g4[:,1],color='m')
    plt.axis([-7, 7, -7, 7])
    plt.show()
    '''

    #Append all samples to create X
    X = np.vstack([g1,g2,g3,g4])

    #Create Y values for samples in X
    Y = np.vstack([0]*Nclass + [1]*Nclass + [2]*Nclass + [3]*Nclass)

    #Number of samples
    N = len(Y)

    #Indicator Variable
    T = np.zeros((N,K))

    #One-Hot Encoding for Y
    for i in range(N):
        T[i, Y[i]] = 1

    #One more scatter plot
    plt.scatter(X[:,0], X[:,1], c=Y, s=80, alpha=0.8)
    plt.axis([-7, 7, -7, 7])
    plt.show()

    #random initialization for weights and bias
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    #learning rate
    eta = 1e-2

    #epsilon
    epsilon = 0.5

    #number of iterations
    num_iter = 250

    #array to store cost per iteration
    costs = []

    #learning happens here for spesified ammount
    for epoch in range(num_iter):
        output, hidden = forward(X, W1, b1, W2, b2)
        c = -1 * cost(T, output)
        #if epoch % 50 == 0:
        #    P = np.argmax(output, axis=1)
        #    r = classification_rate(Y, P)
        #    print("cost: ", c, "classification rate: ", r)
        if c < epsilon:
            break
        costs.append(c)

        W2 += eta * derivative_w2(hidden, T, output)
        b2 += eta * derivative_b2(T, output)
        W1 += eta * derivative_w1(X, hidden, T, output, W2)
        b1 += eta * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

    P_Y_given_X, K = forward(X, W1, b1, W2, b2)
    P = np.argmax(P_Y_given_X, axis=1)

    assert(len(P) == len(Y))

    #One last check for correct clasification rate for samples
    print "Final Classification Rate for Dataset: ", classification_rate(Y,P)

    #Print Confusion Table
    print "Confusion Matrix: "
    confusion_table = pd.crosstab(Y.flatten(),P,margins=True)
    print(confusion_table)

    #Generate grid for contour plot
    x_cont = y_cont = np.arange(-7.0,7.01,0.01)
    xx_cont, yy_cont = np.meshgrid(x_cont,y_cont)
    coordinates = np.c_[xx_cont.ravel(),yy_cont.ravel()]

    #calculate results for grid generated
    Z_Y_given_X,R = forward(coordinates, W1, b1, W2, b2)
    Z = np.argmax(Z_Y_given_X, axis=1)
    Z = Z.reshape(xx_cont.shape)

    #draw plot
    plt.contourf(xx_cont, yy_cont, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=Y, s=80, alpha=0.7)

    #find incorrectly classificatied samples
    E = Y.flatten() != P
    e = np.where( E==True )
    Q = []
    for i in e:
        Q.append(X[i])
    Q = np.array(Q)
    Q = Q[0,:]

    plt.scatter(Q[:,0],Q[:,1],color='r', s=100, alpha= 0.9, marker='^')
    #plt.scatter(Q[:,0], Q[:,1], c='k', s=100, alpha=0.9)

    plt.axis([-7, 7, -7, 7])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == '__main__':
    main()
