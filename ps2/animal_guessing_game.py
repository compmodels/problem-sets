from itertools import product as iproduct
import os
import sys
import numpy as np


def _run(n, query, stream):
    iters = range(n)
    hypothesis_sizes = [2, 4, 8, 16, 32]

    data = dict(np.load('data/50animals.npz'))

    trials = list(iproduct(iters, range(len(hypothesis_sizes))))
    np.random.shuffle(trials)

    trial_data = np.empty((len(iters), len(hypothesis_sizes)), dtype=int)
    for (i, j) in trials:
        hyp_size = hypothesis_sizes[j]
        animal_choices = np.arange(len(data['animal_names']))
        np.random.shuffle(animal_choices)
        animal_choices = animal_choices[:hyp_size]

        animal_index = animal_choices[np.random.randint(hyp_size)]
        animal_name = data['animal_names'][animal_index]
        feature_indices = np.nonzero(data['animal_features'][animal_index])[0]
        np.random.shuffle(feature_indices)

        features = list(data['feature_names'][feature_indices])
        animals = list(data['animal_names'][animal_choices])

        guessed = False
        num_guesses = 0
        stream.write("-" * 70 + "\n")
        while not guessed:
            animal_guess = query(features[:2 + num_guesses], animals)
            if animal_guess == animal_name:
                stream.write("Correct!\n")
                guessed = True
            else:
                stream.write("Sorry, try again.\n")

            stream.write("\n")
            num_guesses += 1

        trial_data[i, j] = num_guesses

    return trial_data


def play():
    if os.path.exists('data/my_trial_data.npy'):
        print("Loading existing data from 'data/my_trial_data.npy'")
        trial_data = np.load('data/my_trial_data.npy')
        return trial_data

    def query(features, animals):
        print("Features: {}".format(features))
        print("Animals: {}".format(animals))
        animal_guess = input("Your guess: ")
        while animal_guess not in animals:
            print("Invalid guess '{}' (did you make a typo?)".format(
                animal_guess))
            animal_guess = input("Your guess: ")
        return animal_guess

    trial_data = _run(3, query, sys.stdout)

    print("Saving trial data to 'data/my_trial_data.npy'")
    np.save('data/my_trial_data.npy', trial_data)

    return trial_data


def model(guess):
    data = dict(np.load('data/50animals.npz'))
    feature_names = list(data['feature_names'])
    animal_names = list(data['animal_names'])

    animal_features = {}
    for i, animal in enumerate(animal_names):
        animal_features[animal] = set([feature_names[j] for j, f in enumerate(data['animal_features'][i]) if f])

    def query(features, animals):
        af = {a: animal_features[a] for a in animals}
        animal_guess = guess(set(features), af)
        return animal_guess

    stream = open(os.devnull, 'w')
    try:
        trial_data = _run(500, query, stream)
    finally:
        stream.close()

    print("Saving trial data to 'data/model_trial_data.npy'")
    np.save('data/model_trial_data.npy', trial_data)

    return trial_data
