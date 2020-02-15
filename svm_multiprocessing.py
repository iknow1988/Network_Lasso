import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import time
from sklearn import svm
from multiprocessing import Pool
np.random.seed(2)
from sklearn.metrics.pairwise import euclidean_distances
import sys
import os

def generateSyntheticData(example):
    if(example == 1):
        sizeOptVar = 51
        group = 20
        trainDataSize = 25000
        testDataSize = 10000
        a_true_group = np.random.randn(group, sizeOptVar)
        x_train = np.random.randn(trainDataSize, sizeOptVar)
        y_train = np.zeros((trainDataSize, 1))               
        x_test = np.random.randn(testDataSize, sizeOptVar)
        y_test = np.zeros((testDataSize, 1))
        v_noise_train = np.random.randn(trainDataSize, 1)
        v_noise_test = np.random.randn(testDataSize, 1)
        
        for i in range(trainDataSize):
            a_part = a_true_group[ i / (trainDataSize / group), :]
            x_train[i, sizeOptVar - 1] = 1
            y_train[i] = np.sign([np.dot(a_part.transpose(), x_train[i, 0:sizeOptVar]) + v_noise_train[i]])
        
        for i in range(testDataSize):
            a_part = a_true_group[ i / (testDataSize / group), :]
            x_test[i, sizeOptVar - 1] = 1
            y_test[i] = np.sign([np.dot(a_part.transpose(), x_test[i, 0:sizeOptVar]) + v_noise_test[i]])
        
        return x_train, y_train, x_test, y_test, a_true_group
    else:
        sizeOptVar = 20
        trainDataSize = 1000
        testDataSize = trainDataSize
        DENSITY = 0.2
        beta_true = np.random.randn(sizeOptVar, 1)
        idxs = np.random.choice(range(sizeOptVar), int((1 - DENSITY) * sizeOptVar), replace=False)
        for idx in idxs:
            beta_true[idx] = 0
        offset = 0
        sigma = 45
        x_train = np.random.normal(0, 5, size=(trainDataSize, sizeOptVar))
        y_train = np.sign(x_train.dot(beta_true) + offset + np.random.normal(0, sigma, size=(trainDataSize, 1)))
        x_test = np.random.normal(0, 5, size=(testDataSize, sizeOptVar))
        y_test = np.sign(x_test.dot(beta_true) + offset + np.random.normal(0, sigma, size=(testDataSize, 1)))
        
        return x_train, y_train, x_test, y_test

def librarySVM(x_train, y_train, x_test, y_test):
    print "fitting model"
    start_time = time.time()
    model = svm.SVC(kernel='linear', C=1, gamma=1, cache_size=1000, verbose=0) 
    model.fit(x_train, y_train.ravel())
    print("--- %s seconds ---" % (time.time() - start_time))
    print "learning model finished"
    print model.score(x_train, y_train)
    print model.score(x_test, y_test)
    print "finish"

def solveOptimizationProblem(data):  
    c = 0.75
    x_train = data[:, 0:data.shape[1] - 1]
    y_train = data[:, data.shape[1] - 1:data.shape[1]]
    a = Variable(x_train.shape[1])
    
    epsil = Variable(x_train.shape[0], 1)
    constraints = [epsil >= 0]
    g = c * norm(epsil, 1)
    for i in range(x_train.shape[1]):
        g = g + 0.5 * square(a[i])
    for i in range(x_train.shape[0]):
        constraints = constraints + [y_train[i] * (x_train[i] * a) >= 1 - epsil[i]]
     
    objective = Minimize(g)
    prob = Problem(objective, constraints)
    result = prob.solve()
    return a.value, result

def main():
    (x_train, y_train, x_test, y_test, a_true) = generateSyntheticData(1)
    trainingData = np.concatenate((x_train, y_train), axis=1)

    x_train = trainingData[:, 0:51]
    y_train = trainingData[:, 51:52]

    sizeOptVar = x_train.shape[1]
    trainDataSize = x_train.shape[0]
    testDataSize = x_test.shape[0]
    
    getLibraryResult = 0
    if (getLibraryResult == 1):
        librarySVM(x_train, y_train, x_test, y_test)
    plot1 = list()
    plot2 = list()
    plot3 = list()
    plot4 = list()
    for nodes in range(80, 25001):
        if ((25000 % nodes == 0)and(1000 % nodes == 0)):
            if not os.path.exists("SVM/Features"):
                os.makedirs("SVM/Features")
            if not os.path.exists("SVM/Similarities"):
                os.makedirs("SVM/Similarities")
            if not os.path.exists("SVM/Accuracies"):
                    os.makedirs("SVM/Accuracies")
            if not os.path.exists("SVM/Time"):
                os.makedirs("SVM/Time")
            print "For model = ", nodes
            
            const = Parameter(sign="positive")
            const.value = 0
            a_estimated = Variable(sizeOptVar)
            v = Variable()
            
            start_time = time.time()
            data = np.transpose(np.concatenate((np.array(np.array_split(x_train, nodes)), np.array(np.array_split(y_train, nodes))), axis=2))
            pool = Pool(4)
            values = pool.map(solveOptimizationProblem, data.transpose())
            a_predicted = np.array(np.array(values)[:, 0].tolist()).transpose()[0]
            timeElapsed = time.time() - start_time
            print "--- %s seconds ---",timeElapsed
            a_true = a_true.transpose()

            # Get accuracy
            (right, total) = (0, trainDataSize)
            for i in range(nodes):
                a = a_predicted[:, i]
                for j in range(trainDataSize / nodes):
                    pred = np.sign([np.dot(a.transpose(), x_train[i * trainDataSize / nodes + j])])
                    if(pred == y_train[i * trainDataSize / nodes + j]):
                        right = right + 1
            trainingAccuracy = right / float(total)
            print "\t Training Accuracy = ", trainingAccuracy
            
            (right, total) = (0, testDataSize)
            for i in range(nodes):
                a = a_predicted[:, i]
                for j in range(testDataSize / nodes):
                    pred = np.sign([np.dot(a.transpose(), x_test[i * testDataSize / nodes + j])])
                    if(pred == y_test[i * testDataSize / nodes + j]):
                        right = right + 1
            testingAccuracy = right / float(total)
            print "\t Testing Accuracy = ", testingAccuracy
        
            models = np.zeros((sizeOptVar,1000))
            eachNodeContains = 1000/nodes
            for i in range(1000):
                models[:,i] = a_predicted[:,i / eachNodeContains]
                
            plt.imshow(models, aspect='auto', interpolation='none', origin='lower')
            plt.colorbar()
            plt.savefig("SVM/Features/"+ str(nodes) +".png")
            plt.clf()
            
            similarities = euclidean_distances(models.transpose())
            plt.imshow(similarities[0:50,0:50], aspect='auto', interpolation='none', origin='lower', cmap='gray')
            plt.colorbar()
            plt.savefig("SVM/Similarities/"+ str(nodes) +".png")
            plt.show()
            
            plot1.append(trainingAccuracy) 
            plot2.append(testingAccuracy)
            plot3.append(timeElapsed)
            plot4.append(nodes)
            
            pl1 = np.array(plot1)
            pl2 = np.array(plot2)
            pl4 = np.array(plot4)
            plt.plot(pl4, pl1, label="Training Data")
            plt.plot(pl4, pl2, label="Testing Data")
            plt.xlabel('Nodes')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig("SVM/Accuracies/"+ str(nodes) +".png")
            plt.clf()
            
            pl3 = np.array(plot3)
            plt.plot(pl4, pl3)
            plt.xlabel('Nodes')
            plt.ylabel('Elapsed Time')
            plt.savefig("SVM/Time/"+ str(nodes) +".png")
            plt.clf()
            
            pool.close()
            pool.join()
if __name__ == '__main__':
    main()
