import networkx as nx
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
import time
import sys
import multiprocessing
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.ticker import MaxNLocator
from networkx.classes.function import neighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn.preprocessing import scale
np.random.seed(2)

# class Logger(object):
#     def __init__(self):
#         self.terminal = sys.stdout
#         self.log = open("console.log", "a")
# 
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)  
# 
#     def flush(self):
#         #this flush method is needed for python 3 compatibility.
#         #this handles the flush command by doing nothing.
#         #you might want to specify some extra behavior here.
#         pass
# Synthetic Data input variables
nodes = 1000
partitions = 20
sizePart = nodes / partitions
samePartitionEdgeProbability = 0.5
diffPartitionEdgeProbability = 0.01
sizeOptimizationVariable = 51
trainSetSizePerNode = 25
testSetSizePerNode = 10

# Generates the Graph
def generateGraph():
    G1 = nx.Graph()
    for i in range(nodes):
        G1.add_node(i)
    correctedges = 0
    for NI in G1.nodes():
        for NI2 in G1.nodes():
            if(NI < NI2):
                if ((NI / sizePart) == (NI2 / sizePart)):
                    # Same partition, edge w.p 0.5
                    if(np.random.random() >= 1 - samePartitionEdgeProbability):
                        G1.add_edge(NI, NI2, weight=1)
                        correctedges = correctedges + 1
                else:
                    if(np.random.random() >= 1 - diffPartitionEdgeProbability):
                        G1.add_edge(NI, NI2, weight=1)
    
    return G1

# Generates Synthetic data
def generateSyntheticData(G1):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    a_true = np.random.randn(sizeOptimizationVariable, partitions)
    v = np.random.randn(trainSetSizePerNode, nodes)
    vtest = np.random.randn(testSetSizePerNode, nodes)
    
    trainingSet = np.random.randn(trainSetSizePerNode * (sizeOptimizationVariable + 1), nodes)  # First all the x_train, then all the y_train below it
    a_true_nodes = np.zeros((sizeOptimizationVariable, nodes))
    for i in range(trainSetSizePerNode):
        trainingSet[(i + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
    for i in range(nodes):
        a_part = a_true[:, i / sizePart]
        a_true_nodes[:, i] = a_part
        for j in range(trainSetSizePerNode):
            trainingSet[trainSetSizePerNode * sizeOptimizationVariable + j, i] = np.sign([np.dot(a_part.transpose(), trainingSet[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i]) + v[j, i]])

    (x_test, y_test) = (np.random.randn(testSetSizePerNode * sizeOptimizationVariable, nodes), np.zeros((testSetSizePerNode, nodes)))
    for i in range(testSetSizePerNode):
        x_test[(i + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
    for i in range(nodes):
        a_part = a_true[:, i / sizePart]
        for j in range(testSetSizePerNode):
            y_test[j, i] = np.sign([np.dot(a_part.transpose(), x_test[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i]) + vtest[j, i]])

    return trainingSet, x_test, y_test, a_true, a_true_nodes.transpose()

# Gets the neighboring Models information of every nodes
def getNeighborsModelParameters(G1, maxdeg, u , z):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()

    neighbors = np.zeros(((2 * sizeOptimizationVariable + 1) * maxdeg, nodes))
    edgenum = 0
    numSoFar = {}
    for EI in G1.edges(data=True):
        if not EI[0] in numSoFar:
            numSoFar[EI[0]] = 0
        sourceNode = EI[0]
        neighborIndex = numSoFar[EI[0]]
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1), sourceNode] = EI[2]['weight']
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + 1:neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1), sourceNode] = u[:, 2 * edgenum] 
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1):(neighborIndex + 1) * (2 * sizeOptimizationVariable + 1), sourceNode] = z[:, 2 * edgenum]
        numSoFar[EI[0]] = numSoFar[EI[0]] + 1

        if not EI[1] in numSoFar:
            numSoFar[EI[1]] = 0
        sourceNode = EI[1]
        neighborIndex = numSoFar[EI[1]]
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1), sourceNode] = EI[2]['weight']
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + 1:neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1), sourceNode] = u[:, 2 * edgenum + 1] 
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1):(neighborIndex + 1) * (2 * sizeOptimizationVariable + 1), sourceNode] = z[:, 2 * edgenum + 1]
        numSoFar[EI[1]] = numSoFar[EI[1]] + 1
        
        edgenum = edgenum + 1
    
    return neighbors

# solves optimization problem
def solveX(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    sizeData = int(data[data.size - 4])
    trainingSetSize = int(data[data.size - 5])
    c = 0.75
    x = data[0:optimizationVariableSize]
    trainingData = data[optimizationVariableSize:(optimizationVariableSize + sizeData)]
    neighbors = data[(optimizationVariableSize + sizeData):data.size - 6]
    x_train = trainingData[0:trainingSetSize * optimizationVariableSize]
    y_train = trainingData[trainingSetSize * optimizationVariableSize: trainingSetSize * (optimizationVariableSize + 1)]
    
    a = Variable(optimizationVariableSize, 1)
    epsil = Variable(trainingSetSize, 1)
    constraints = [epsil >= 0]
    
    g = c * norm(epsil, 1)
    for i in range(optimizationVariableSize - 1):
        g = g + 0.5 * square(a[i])
    for i in range(trainingSetSize):
        temp = np.asmatrix(x_train[i * optimizationVariableSize:(i + 1) * optimizationVariableSize])
        constraints = constraints + [y_train[i] * (temp * a) >= 1 - epsil[i]]
    
    f = 0
    for i in range(neighbors.size / (2 * optimizationVariableSize + 1)):
        weight = neighbors[i * (2 * optimizationVariableSize + 1)]
        if(weight != 0):
            u = neighbors[i * (2 * optimizationVariableSize + 1) + 1:i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1)]
            z = neighbors[i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1):(i + 1) * (2 * optimizationVariableSize + 1)]
            f = f + rho / 2 * square(norm(a - z + u))
    
    objective = Minimize(50 * g + 50 * f)
    p = Problem(objective, constraints)
    result = p.solve()
    
    return a.value, g.value

# Updates Z values
def solveZ(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    weight = data[data.size - 4]
    x1 = data[0:optimizationVariableSize]
    x2 = data[optimizationVariableSize:2 * optimizationVariableSize]
    u1 = data[2 * optimizationVariableSize:3 * optimizationVariableSize]
    u2 = data[3 * optimizationVariableSize:4 * optimizationVariableSize]
    a = x1 + u1
    b = x2 + u2
    
    (z1, z2) = (0, 0)
    theta = max(1 - lamb * weight / (rho * LA.norm(a - b) + 0.000001), 0.5)  # So no divide by zero error
    z1 = theta * a + (1 - theta) * b
    z2 = theta * b + (1 - theta) * a
          
    znew = np.matrix(np.concatenate([z1, z2]))
    znew = znew.reshape(2 * optimizationVariableSize, 1)
    return znew

# Updates dual variable
def solveU(data):
    length = data.size
    u = data[0:length / 3]
    x = data[length / 3:2 * length / 3]
    z = data[(2 * length / 3):length]
    
    return u + (x - z)

# Initializes ADMM algorithm
def initializeADMM(G1):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    counter = 0
    A = np.zeros((2 * edges, nodes))
    for EI in G1.edges():
        A[2 * counter, EI[0]] = 1
        A[2 * counter + 1, EI[1]] = 1
        counter = counter + 1
    (sqn, sqp) = (math.sqrt(nodes * sizeOptimizationVariable), math.sqrt(2 * sizeOptimizationVariable * edges))
    x = np.zeros((sizeOptimizationVariable, nodes))
    u = np.zeros((sizeOptimizationVariable, 2 * edges))
    z = np.zeros((sizeOptimizationVariable, 2 * edges))

    return A, sqn, sqp, x, u, z  

# Runs ADMM on graph
def runADMM(G1, lamb, rho, x, u, z, a, A, sqn, sqp, maxProcesses, eabs, erel, admmMaxIteration, directory, x_train, y_train, x_test, y_test, a_true_nodes):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    maxdeg = max(G1.degree().values());
    sizeData = a.shape[0]
    # stopping Criterion
    (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)   
    
    # Run ADMM
    iters = 1
    pool = Pool(maxProcesses)
    while(r > epri or s > edual):
        print "\t \t At Iteration = ", iters
        
        start_time = time.time()
        
        # x-update
        neighbors = getNeighborsModelParameters(G1, maxdeg, u , z)
        params = np.tile([trainSetSizePerNode, sizeData, rho, lamb, sizeOptimizationVariable], (nodes, 1)).transpose()
        temp = np.concatenate((x, a, neighbors, params), axis=0)
        values = pool.map(solveX, temp.transpose())
        newx = np.array(values)[:, 0].tolist()
        x = np.array(newx).transpose()[0]

        # z-update
        ztemp = z.reshape(2 * sizeOptimizationVariable, edges, order='F')
        utemp = u.reshape(2 * sizeOptimizationVariable, edges, order='F')
        xtemp = np.zeros((sizeOptimizationVariable, 2 * edges))
        counter = 0
        weightsList = np.zeros((1, edges))
        for EI in G1.edges(data=True):
            xtemp[:, 2 * counter] = np.array(x[:, EI[0]])
            xtemp[:, 2 * counter + 1] = x[:, EI[1]]
            weightsList[0, counter] = EI[2]['weight']
            counter = counter + 1
        xtemp = xtemp.reshape(2 * sizeOptimizationVariable, edges, order='F')
        temp = np.concatenate((xtemp, utemp, ztemp, np.reshape(weightsList, (-1, edges)), np.tile([rho, lamb, sizeOptimizationVariable], (edges, 1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        ztemp = np.array(newz).transpose()[0]
        ztemp = ztemp.reshape(sizeOptimizationVariable, 2 * edges, order='F')
        s = LA.norm(rho * np.dot(A.transpose(), (ztemp - z).transpose()))  # For dual residual
        z = ztemp
        
        # u-update
        (xtemp, counter) = (np.zeros((sizeOptimizationVariable, 2 * edges)), 0)
        for EI in G1.edges(data=True):
            xtemp[:, 2 * counter] = np.array(x[:, EI[0]])
            xtemp[:, 2 * counter + 1] = x[:, EI[1]]
            counter = counter + 1
        temp = np.concatenate((u, xtemp, z), axis=0)
        newu = pool.map(solveU, temp.transpose())
        u = np.array(newu).transpose()
        
        # Stopping criterion - p19 of ADMM paper
        epri = sqp * eabs + erel * max(LA.norm(np.dot(A, x.transpose()), 'fro'), LA.norm(z, 'fro'))
        edual = sqn * eabs + erel * LA.norm(np.dot(A.transpose(), u.transpose()), 'fro')
        r = LA.norm(np.dot(A, x.transpose()) - z.transpose(), 'fro')
         
        print "\t \t \t Iteration ", iters, " took time ", (time.time() - start_time), "seconds", "And Primal residual = ", r, " , Dual Residual = ", s
        similarities = euclidean_distances(x.transpose())
        plt.imshow(similarities, aspect='auto', interpolation='none', origin='lower', cmap='gray')
        plt.colorbar()
        plt.show()
        iters = iters + 1
    
    pool.close()
    pool.join()
    return (x, u, z, iters-1)

def getAccuracy(modelParameters, dataSetSize, featureData, labelData):
    write = np.zeros((nodes, 6))
    (right, total) = (0, dataSetSize * nodes)
    a_pred = modelParameters
    for i in range(nodes):
        write[i][0] = i + 1
        nodeRight = 0
        temp = a_pred[:, i]
        for j in range(dataSetSize):
            pred = np.sign([np.dot(temp.transpose(), featureData[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i])])
            if (int(pred) == -1):
                write[i, 3] = write[i, 3] + 1
            else:
                write[i, 1] = write[i, 1] + 1
            if(pred == labelData[j, i]):
                right = right + 1
                nodeRight = nodeRight + 1
        write[i][5] = nodeRight / float(dataSetSize)
    
    for i in range(nodes):
        ex = labelData[:, i]
        write[i][2] = int(ex[ex == 1].sum())
        write[i][4] = int(ex[ex == -1].sum() * -1) 
    
    
    return right / float(total), write

def runExperiments(G1, lamb, rho, trainingSet, x_test, y_test, a_true, maxProcesses, eabs, erel, admmMaxIteration, directory, a_true_nodes):
    x_train = trainingSet[0:trainSetSizePerNode * sizeOptimizationVariable, :]
    y_train = trainingSet[trainSetSizePerNode * sizeOptimizationVariable: trainSetSizePerNode * (sizeOptimizationVariable + 1), :]
    
    # Initialize ADMM variables
    (A, sqn, sqp, x, u, z) = initializeADMM(G1)
    # run ADMM
    print "\t For lambda = ", lamb
    start_time = time.time()
    (x, u, z, iters) = runADMM(G1, lamb, rho + math.sqrt(lamb), x, u , z, trainingSet, A, sqn, sqp, maxProcesses, eabs, erel, admmMaxIteration, directory, x_train, y_train, x_test, y_test, a_true_nodes)
    print("\t \t ADMM finished in %s seconds" % (time.time() - start_time))
                       
    # Get accuracy
    print "\t \t Result"
    (trainAccuracy, write) = getAccuracy(x, trainSetSizePerNode, x_train, y_train)   
    
    print "\t \t \t Train Data Accuracy = ", trainAccuracy
    (testAccuracy, write) = getAccuracy(x, testSetSizePerNode, x_test, y_test)
    print "\t \t \t Test Data Accuracy = ", testAccuracy
        
    return trainAccuracy, testAccuracy

def main():  
    G1 = generateGraph()
    (trainingSet, x_test, y_test, a_true, a_true_nodes) = generateSyntheticData(G1)
#     print(nx.clustering(G1))
#     nm = LA.norm(a_true_nodes,axis=1)
#     nx.draw(G1,pos=nx.spring_layout(G1),cmap=plt.get_cmap('jet'), node_color=nm,arrows=False)
#     plt.show()
#     plt.clf()
    print "Number of Nodes = ", G1.number_of_nodes(), " , Number of Edges = ", G1.number_of_edges()
    print "Diameter is ", nx.diameter(G1);
    maxProcesses = multiprocessing.cpu_count()
    rho = 0.0001
    eabs = math.pow(10, -2)
    erel = math.pow(10, -3)
    admmMaxIteration = 50

    np.set_printoptions(suppress=True) 
    lambdas = np.arange(0,11,0.1)
    np.set_printoptions(suppress=False)
    for lamb in lambdas:
        print rho + math.sqrt(lamb)
    for lamb in lambdas:
        directory = "Plots2/" + str(lamb)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        (trainAccuracy, testAccuracy) = runExperiments(G1, lamb, rho, trainingSet, x_test, y_test, a_true, maxProcesses, eabs, erel, admmMaxIteration, directory, a_true_nodes)
           
if __name__ == '__main__':
    main()
