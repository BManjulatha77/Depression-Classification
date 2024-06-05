import numpy as np
import time


def OOA(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape[0], X.shape[1]
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)
    ct = time.time()
    for t in range(Max_iterations):  # algorithm iteration
        # update: BEST proposed solution
        Fbest, blocation = np.min(fit), np.argmin(fit)
        if t == 0:
            xbest = X[blocation, :]  # Optimal location
            fbest = Fbest  # The optimization objective function
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation, :]

        for i in range(SearchAgents):
            # Phase 1: POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            fish_position = np.where(fit < fit[i])[0]  # Eq(4)
            if len(fish_position) == 0:
                selected_fish = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_fish = xbest
                else:
                    k = np.random.randint(len(fish_position))
                    selected_fish = X[fish_position[k]]

            I = np.round(1 + np.random.rand())
            X_new_P1 = X[i,:] + np.random.rand(1, 1) * (selected_fish - I * X[i,:])  # Eq(5)
            X_new_P1 = max(X_new_P1, lowerbound);
            X_new_P1 = min(X_new_P1, upperbound);

            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit(i):
                X[i,:]  = X_new_P1
                fit[i] = fit_new_P1
            X_new_P1 = X[i,:]+(lowerbound + np.random.rand(upperbound - lowerbound)) / t
            X_new_P1 = max(X_new_P1, lowerbound)
            X_new_P1 = min(X_new_P1, upperbound)

            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]:
                X[i,:] = X_new_P1

                fit[i] = fit_new_P1
            best_so_far= fbest
        # average = mean(fit)

    Best_score = fbest
    Best_pos = xbest
    OOA_curve = best_so_far


    ct = time.time() - ct
    return Best_score, Best_pos, OOA_curve,ct


