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
np.random.seed(2)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("console_Rho.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
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
    
    plt.imshow(a_true, aspect='auto', interpolation='none', origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, a_true.shape[1], dtype=np.int))
    plt.savefig("Plots_Rho/Original_Grouping.png")
    plt.clf()
    
    plt.imshow(a_true_nodes, aspect='auto', interpolation='none', origin='lower')
    plt.colorbar()
    plt.savefig("Plots_Rho/Original_Grouping_nodes.png")
    plt.clf()
    
    similarities = euclidean_distances(a_true_nodes.transpose())
    plt.imshow(similarities, aspect='auto', interpolation='none', origin='lower', cmap='gray')
    plt.colorbar()
    plt.savefig("Plots_Rho/Original_Grouping_nodes_Similarity.png")
    plt.clf()
    
    return trainingSet, x_test, y_test, a_true

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
def runADMM(G1, lamb, rho, x, u, z, a, A, sqn, sqp, maxProcesses, eabs, erel, admmMaxIteration, directory, x_train, y_train, x_test, y_test):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    maxdeg = max(G1.degree().values());
    sizeData = a.shape[0]
    # stopping Criterion
    (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)   
    
    # Run ADMM
    iters = 1
    pool = Pool(maxProcesses)
    plot1 = list()
    plot2 = list()
    plot3 = list()
    plot4 = list()
    plot5 = list()
    plot6 = list()
    if not os.path.exists(directory+"/Experiments/Features"):
            os.makedirs(directory+"/Experiments/Features")
    if not os.path.exists(directory+"/Experiments/Similarities"):
        os.makedirs(directory+"/Experiments/Similarities")
    if not os.path.exists(directory+"/Experiments/Residuals"):
        os.makedirs(directory+"/Experiments/Residuals")
    if not os.path.exists(directory+"/Experiments/Accuracies"):
        os.makedirs(directory+"/Experiments/Accuracies")
    if not os.path.exists(directory+"/Experiments/Stopping Criterion"):
        os.makedirs(directory+"/Experiments/Stopping Criterion")
    if not os.path.exists(directory+"/Experiments/combined"):
        os.makedirs(directory+"/Experiments/combined")
    while(r > epri or s > edual):
        print "\t \t At Iteration = ", iters
        plot1.append(r) 
        plot2.append(s)
        plot5.append(epri)
        plot6.append(edual)
        
        pl1 = np.array(plot1)
        pl2 = np.array(plot2)
        plt.plot(pl1, label="Primal")
        plt.plot(pl2, label="Dual")
        plt.xlabel('Iteration')
        plt.ylabel('Residuals')
        plt.legend()
        plt.savefig(directory+"/Experiments/Residuals/"+ str(iters) +".png")
        plt.clf()
        
        pl5 = np.array(plot5)
        pl6 = np.array(plot6)
        plt.plot(pl5, label="Primal stopping value")
        plt.plot(pl6, label="Dual stopping value")
        plt.xlabel('Iteration')
        plt.ylabel('Stopping Criterion')
        plt.legend()
        plt.savefig(directory+"/Experiments/Stopping Criterion/"+ str(iters) +".png")
        plt.clf()
        
        plt.plot(pl1, label="Primal")
        plt.plot(pl2, label="Dual")
        plt.plot(pl5, label="Primal stopping value")
        plt.plot(pl6, label="Dual stopping value")
        plt.xlabel('Iteration')
        plt.ylabel('Stopping values')
        plt.legend()
        plt.savefig(directory+"/Experiments/combined/"+ str(iters) +".png")
        plt.clf()
        
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
            
        plt.imshow(x, aspect='auto', interpolation='none', origin='lower')
        plt.colorbar()
        plt.savefig(directory+"/Experiments/Features/" + str(iters) +".png")
        plt.clf()
        
        similarities = euclidean_distances(x.transpose())
        plt.imshow(similarities, aspect='auto', interpolation='none', origin='lower', cmap='gray')
        plt.colorbar()
        plt.savefig(directory+"/Experiments/Similarities/" + str(iters) +".png")
        plt.clf()
        
        (trainAccuracy, write) = getAccuracy(x, trainSetSizePerNode, x_train, y_train)
        (testAccuracy, write) = getAccuracy(x, testSetSizePerNode, x_test, y_test)
        
        plot3.append(trainAccuracy)
        plot4.append(testAccuracy)
        
        pl3 = np.array(plot3)
        pl4 = np.array(plot4)
        plt.plot(pl3, label="Train")
        plt.plot(pl4, label="Test")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(directory+"/Experiments/Accuracies/"+ str(iters) +".png")
        plt.clf()
         
        print "\t \t \t Took time ", (time.time() - start_time), "seconds", "And Primal residual = ", r, " , Dual Residual = ", s, ", ePri =",epri,", eDual=",edual
        print "\t \t \t Training Accuracy =",trainAccuracy,", Testing Accuracy =",testAccuracy

        iters = iters + 1
    
    pool.close()
    pool.join()
    pl1 = np.array(plot1)
    pl2 = np.array(plot2)
    plt.plot(pl1, label="Primal")
    plt.plot(pl2, label="Dual")
    plt.xlabel('Iteration')
    plt.ylabel('Residuals')
    plt.legend()
    plt.savefig(directory + "/Resiudals.png")
    plt.clf()
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

def runExperiments(G1, lamb, rho, trainingSet, x_test, y_test, a_true, maxProcesses, eabs, erel, admmMaxIteration, directory):
    x_train = trainingSet[0:trainSetSizePerNode * sizeOptimizationVariable, :]
    y_train = trainingSet[trainSetSizePerNode * sizeOptimizationVariable: trainSetSizePerNode * (sizeOptimizationVariable + 1), :]
    
    # Initialize ADMM variables
    (A, sqn, sqp, x, u, z) = initializeADMM(G1)
    # run ADMM
    print "\t For Rho = ", rho
    start_time = time.time()
    (x, u, z, iters) = runADMM(G1, lamb, rho + math.sqrt(lamb), x, u , z, trainingSet, A, sqn, sqp, maxProcesses, eabs, erel, admmMaxIteration, directory, x_train, y_train, x_test, y_test)
    print("\t \t ADMM finished in %s seconds" % (time.time() - start_time))
                       
    # Get accuracy
    print "\t \t Result"
    (trainAccuracy, write) = getAccuracy(x, trainSetSizePerNode, x_train, y_train)   
    np.savetxt(directory + "/training.csv", write, header = 'Node,Positive Sample,Estimated Positive Sample,Negative Sample,Estimated Negative Sample,Accuracy', fmt='%.2f', delimiter=",")
    
    print "\t \t \t Train Data Accuracy = ", trainAccuracy
    (testAccuracy, write) = getAccuracy(x, testSetSizePerNode, x_test, y_test)
    np.savetxt(directory + "/testing.csv", write, header = 'Node,Positive Sample,Estimated Positive Sample,Negative Sample,Estimated Negative Sample,Accuracy', fmt='%.2f', delimiter=",")
    print "\t \t \t Test Data Accuracy = ", testAccuracy
        
    plt.imshow(x, aspect='auto', interpolation='none', origin='lower')
    plt.colorbar()
    plt.savefig(directory + "/features.png")
    plt.clf()
    
    similarities = euclidean_distances(x.transpose())
    plt.imshow(similarities, aspect='auto', interpolation='none', origin='lower', cmap='gray')
    plt.colorbar()
    plt.savefig(directory + "/Similarities.png")
    plt.clf()
    
    return trainAccuracy, testAccuracy, iters

def main():
    sys.stdout = Logger()
    G1 = generateGraph()
    
    (trainingSet, x_test, y_test, a_true) = generateSyntheticData(G1)
    print "Number of Nodes = ", G1.number_of_nodes(), " , Number of Edges = ", G1.number_of_edges()
    print "Diameter is ", nx.diameter(G1);
    maxProcesses = multiprocessing.cpu_count()
    rho = 0.0001
    admmMaxIteration = 50
    np.set_printoptions(suppress=True)
    lambdas = np.arange(1.1,1.5,0.1)
    np.set_printoptions(suppress=False)
    plot1 = list()
    plot2 = list()
    plot3 = list()
    plot4 = list()
    plot5 = list()
    iters = 0
    for i in range(6):
        eabs = math.pow(10, -2)
        erel = math.pow(10, -2)
        rho = math.pow(10, i*(-1))
        lamb = 1.1
        directory = "Plots_Rho/" + str(rho)
        if not os.path.exists(directory):
            os.makedirs(directory)
        start_time = time.time()
        (trainAccuracy, testAccuracy, iterations) = runExperiments(G1, lamb, rho, trainingSet, x_test, y_test, a_true, maxProcesses, eabs, erel, admmMaxIteration, directory)
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        if not os.path.exists("Plots_Rho/Accuracies"):
            os.makedirs("Plots_Rho/Accuracies")
        if not os.path.exists("Plots_Rho/Time"):
            os.makedirs("Plots_Rho/Time")
        if not os.path.exists("Plots_Rho/Iterations"):
            os.makedirs("Plots_Rho/Iterations")
                    
        plot1.append(trainAccuracy)
        plot2.append(testAccuracy)
        plot3.append(iterations)
        plot4.append(minutes)
        plot5.append(rho)
        pl1 = np.array(plot1)
        pl2 = np.array(plot2)
        pl3 = np.array(plot3)
        pl4 = np.array(plot4)
        pl5 = np.array(plot5)
        
        plt.plot(pl5, pl1, label="Train")
        plt.plot(pl5, pl2, label="Test")
        plt.xlabel('rho')
        plt.ylabel('Prediction Accuracy')
        plt.legend()
        plt.savefig("Plots_Rho/Accuracies/"+ str(rho) +".png")
        plt.clf()
        
        plt.plot(pl5, pl4)
        plt.xlabel('rho')
        plt.ylabel('Time (minutes)')
        plt.savefig("Plots_Rho/Time/"+ str(rho) +".png")
        plt.clf()
        
        plt.plot(pl5, pl3)
        plt.xlabel('rho')
        plt.ylabel('Iterations')
        plt.savefig("Plots_Rho/Iterations/"+ str(rho) +".png")
        plt.clf()
        
        iters = iters + 1
         
if __name__ == '__main__':
    main()
