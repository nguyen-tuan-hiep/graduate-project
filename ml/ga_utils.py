import numpy as np
from random import randint
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
import joblib


def initilization_of_population(size, n_feat):
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool_)
        chromosome[:int(0.3*n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(best_model, best_score, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, population):
    scores = []

    for chromosome in tqdm(population, desc='Training models with GA...'):
        model = logmodel

        model.fit(X_train.iloc[:, chromosome], y_train)
        predictions = model.predict(X_test.iloc[:, chromosome])
        scores.append(r2_score(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print('Score:', scores[-1], 'MSE', mse, 'Chromosome:', chromosome)


        if scores[-1] > best_score['r2']:
            best_score['mae'] = mae
            best_score['mse'] = mse
            best_score['rmse'] = rmse
            best_score['r2'] = scores[-1]
            best_chromo_all = chromosome
            best_model = clone(model)


    scores, population = np.array(scores), np.array(population)

    inds = np.argsort(scores)

    return list(scores[inds][::-1]), list(population[inds][::-1]), best_score, best_chromo_all, best_model


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0, len(pop_after_sel), 2):
        new_par = []
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i+1]
        new_par = np.concatenate(
            (child_1[:len(child_1)//2], child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(best_score, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, size, n_feat, n_parents, mutation_rate, n_gen):
    population_nextgen = initilization_of_population(size, n_feat)
    best_model = None
    for i in range(n_gen):
        scores, pop_after_fit, best_score, best_chromo_all, best_model = fitness_score(best_model, best_score, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, population_nextgen)
        print('Best score in generation', i+1, ':', scores[0])

        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)

    return best_model, best_score, best_chromo_all
