import time
import numpy as np
from mpmath import eps, atanh, rand, log10
from random import uniform

def ESMA(X, fobj, lb, ub, Max_iter):
    N, dim = X.shape[0], X.shape[1]
    bestPositions = np.zeros((1, dim))
    Destination_fitness = np.inf
    AllFitness = np.inf * np.ones((N, 1))
    weight = np.ones((N, dim))
    Convergence_curve = np.zeros((1, Max_iter))
    it = 0
    lb = np.multiply(np.ones((1, dim)), lb)
    ub = np.multiply(np.ones((1, dim)), ub)
    z = 0.03
    # Main loop
    ct = time.time()
    while it <= Max_iter:
        for i in np.arange(1, N + 1).reshape(-1):
            # Check if solutions go outside the search space and bring them back
            Flag4ub = X[i,:] > ub
            Flag4lb = X[i,:] < lb
            X[i, :] = (np.multiply(X[i,:], (not (Flag4ub + Flag4lb)))) + np.multiply(ub, Flag4ub) + np.multiply(lb,Flag4lb)
            AllFitness[i] = fobj[X[i,:]]
            ct = time.time()
            # sort the fitness
            SmellOrder, SmellIndex = np.sort(AllFitness)
            bestFitness = SmellOrder(1)
            worstFitness = SmellOrder(N)
            bestPositions2 = X[SmellIndex(2),:]
            bestPositions3 = X[SmellIndex(3),:]
            bestPositions4 = X[SmellIndex(4),:]
            S = bestFitness - worstFitness + eps
            # calculate the fitness weight of each slime mold
            for i in np.arange(1, N + 1).reshape(-1):
                for j in np.arange(1, dim + 1).reshape(-1):
                    if i <= (N / 2):
                        weight[SmellIndex[i], j] = 1 + np.random.rand() * log10((bestFitness - SmellOrder(i)) / (S) + 1)
                    else:
                        weight[SmellIndex[i], j] = 1 - np.random.rand() * log10((bestFitness - SmellOrder(i)) / (S) + 1)
            # update the best fitness value and best position
            if bestFitness < Destination_fitness:
                bestPositions = X[SmellIndex(1),:]
                Destination_fitness = bestFitness
            avgPositions = (bestPositions + bestPositions2 + bestPositions3 + bestPositions4) / 4
            C_pool = np.array([[bestPositions], [bestPositions2], [bestPositions3], [bestPositions4], [avgPositions]])
            a = atanh(- (it / Max_iter) + 1)
            b = 1 - it / Max_iter
            # Update the Position of search agents
            for i in np.arange(1, N + 1).reshape(-1):
                if rand < z:
                    X[i, :] = (ub - lb) * rand + lb
                else:
                    p = np.tanh(np.abs(AllFitness[i] - Destination_fitness))
                    vb = uniform(- a, a, 1, dim)
                    vc = uniform(- b, b, 1, dim)
                    for j in np.arange(1, dim + 1).reshape(-1):
                        r = np.random.rand()
                        Ceq = C_pool[np.random.randint(C_pool.shape[1 - 1]),:]
                        A = np.random.randint(np.array([1, N]))
                        if r < p:
                            X[i, j] = bestPositions[j] + vb[j] * (weight[i, j] * Ceq[j] - X(A, j))
                        else:
                            X[i, j] = vc[j] * X(i, j)
            Convergence_curve[it] = Destination_fitness
            it = it + 1
    ct = time.time() - ct
    return Destination_fitness, bestPositions, Convergence_curve,ct