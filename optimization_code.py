import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import __main__
np.random.seed(5)

def main():
    # Initialize some data with gaussian random noise
    f, axarr = plt.subplots(3, 2)
    
    x = np.arange(40)
    y = 0.3 * x + 5 + np.random.standard_normal(40)
    axarr[0,0].set_title('Some data with gaussian random noise')
    axarr[0,0].scatter(x, y)
    #plt.show()
    
    # Perturb it!
    for i in xrange(40):
        if np.random.random() < 0.1:
            y[i] += 10
    axarr[1,0].scatter(x, y)
    axarr[1,0].set_title('Some perturbance')
    #plt.show()
    
    # Try linear regression
    w = cvxpy.Variable(); b = cvxpy.Variable()
    obj = 0
    for i in xrange(40):
        obj += (w * x[i] + b - y[i]) ** 2
    cvxpy.Problem(cvxpy.Minimize(obj), []).solve()
    w = w.value; b = b.value
    axarr[2,0].set_title('Linear Regression')
    axarr[2,0].scatter(x, y)
    axarr[2,0].plot(x, w * x + b)
    
    # Create a classification dataset
    x1 = np.random.normal(2, 1, (2, 40))
    x2 = np.random.normal(-2, 1, (2, 40))
    axarr[0,1].set_title('Classification dataset')
    axarr[0,1].scatter(x1[0, :], x1[1, :], color='blue')
    axarr[0,1].scatter(x2[0, :], x2[1, :], color='green')
    
    # Code up an SVM, no liblinear/svmlight needed.
    w = cvxpy.Variable(2); b = cvxpy.Variable()
    obj = 0
    for i in xrange(40):
        obj += cvxpy.pos(1 - (w.T * x1[:, i] + b))
        obj += cvxpy.pos(1 + (w.T * x2[:, i] + b))
    cvxpy.Problem(cvxpy.Minimize(obj), []).solve()
    x = np.arange(-6, 4)
    y = -(w.value[0, 0] * x + b.value) / w.value[1, 0]
    axarr[1,1].set_title('SVM')
    axarr[1,1].plot(x, y, color='red')
    axarr[1,1].scatter(x1[0, :], x1[1, :], color='blue')
    axarr[1,1].scatter(x2[0, :], x2[1, :], color='green')
    
    plt.show()
    
if __name__ == '__main__':
    main()