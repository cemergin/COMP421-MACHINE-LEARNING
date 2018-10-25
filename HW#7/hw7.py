import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

def dif(u,v):
    d = u - v
    return d.dot(d)

def cost(X, R, M):
    cost = 0
    for k in xrange(len(M)):
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def KNN(X, K , iter=20, beta=1, show_plots=True):
    N, D = X.shape
    M = np.zeros((K,D))
    exp = np.empty((N,K))

    for k in xrange(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(iter)
    for i in xrange(iter):
        for k in xrange(K):
            for n in xrange(N):
                exp[n,k] = np.exp(-beta*dif(M[k], X[n]))

        R = exp / exp.sum(axis = 1, keepdims= True)

        for k in xrange(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()

        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 10e-5:
                break
    if show_plots:
        plt.plot(costs)
        plt.title("Costs vs. Iterations for KNN")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors,s=60, alpha=0.9)
        plt.title("KNN")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    return M, R

def GMM(X, K, iter = 20, show_plots=True, smoothing=10e-4, Mi=None, Ri=None, mu=None, cov=None):
    N, D = X.shape
    M = np.zeros((K, D))
    C = np.zeros((K, D, D))
    pi = np.ones(K) / K

    if Ri == None:
        R = np.zeros((N, K))
    else:
        R = Ri

    for k in xrange(K):
        M[k] = X[np.random.choice(N)]
        C[k] = np.eye(D)

    if Mi != None:
        M = Mi

    costs = np.zeros(iter)
    wpdfs = np.zeros((N, K))
    for i in xrange(iter):
        for k in xrange(K):
            for n in xrange(N):
                wpdfs[n,k] = pi[k]*multivariate_normal.pdf(X[n], M[k], C[k])

        for k in xrange(K):
            for n in xrange(N):
                R[n,k] = wpdfs[n,k] / wpdfs[n,:].sum()

        for k in xrange(K):
                Nk = R[:,k].sum()
                pi[k] = Nk / N
                M[k] = R[:,k].dot(X) / Nk
                C[k] = np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in xrange(N)) / Nk + np.eye(D)*smoothing

                costs[i] = np.log(wpdfs.sum(axis=1)).sum()
                if i > 0:
                    if np.abs(costs[i] - costs[i-1]) < 0.1:
                        break
    if show_plots:

        plt.plot(costs)
        plt.title("Costs vs. Number of Iterations for GMM")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors,s=60, alpha=0.9)
        plt.title("GMM")
        plt.xlabel('x1')
        plt.ylabel('x2')

        ax = plt.subplot(111)
        for i in xrange(K):
            m = M[i]
            c = C[i]

            lamb, v = np.linalg.eig(c)
            lamb = np.sqrt(lamb)
            nstd = 2

            vals, vecs = eigsorted(c)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)

            ell = Ellipse(xy=(m[0],m[1]),
              width=w, height=h,
              angle=theta, color='black', ls='--')

            ell.set_facecolor('none')
            ax.add_artist(ell)

        if mu != None and cov != None:
            for i in xrange(K):
                m = mu[i]
                c = cov[i]

                lamb, v = np.linalg.eig(c)
                lamb = np.sqrt(lamb)
                nstd = 2

                vals, vecs = eigsorted(c)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)

                ell = Ellipse(xy=(m[0],m[1]),
                  width=w, height=h,
                  angle=theta, color='red')

                ell.set_facecolor('none')
                ax.add_artist(ell)

        plt.show()

    #print "pi:", pi
    print "means:"
    print M
    #print "covariances:", C
    return R

def get_Data():

        cov1 = np.array([[0.8,-0.6],[-0.6,0.8]])
        cov2 = np.array([[0.8,0.6],[0.6,0.8]])
        cov3 = np.array([[0.8,-0.6],[-0.6,0.8]])
        cov4 = np.array([[0.8,0.6],[0.6,0.8]])
        cov5 = np.array([[1.6,0.0],[0.0,1.6]])

        cov = [cov1,cov2,cov3,cov4,cov5]
        cov = np.asarray(cov)

        mu1 = np.array([2.5,2.5])
        mu2 = np.array([-2.5,2.5])
        mu3 = np.array([-2.5,-2.5])
        mu4 = np.array([2.5,-2.5])
        mu5 = np.array([0.0,0.0])

        mu = [mu1,mu2,mu3,mu4,mu5]
        mu = np.asarray(mu)

        x1 = np.random.multivariate_normal(mean = mu1, cov=cov1,size=50)
        x2 = np.random.multivariate_normal(mean = mu2, cov=cov2,size=50)
        x3 = np.random.multivariate_normal(mean = mu3, cov=cov3,size=50)
        x4 = np.random.multivariate_normal(mean = mu4, cov=cov4,size=50)
        x5 = np.random.multivariate_normal(mean = mu5, cov=cov5,size=100)

        X = np.vstack([x1,x2,x3,x4,x5])

        return X, mu,cov

def main():

    X, mu, cov = get_Data()
    plt.scatter(X[:,0], X[:,1])
    plt.axis([-6, 6, -6, 6])
    plt.title('Initial Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    K = 5
    M, R = KNN(X,K, iter=2, show_plots=False)
    R = GMM(X, K, Mi=M, Ri=R, mu=mu, cov=cov)

if __name__ == '__main__':
    main()
