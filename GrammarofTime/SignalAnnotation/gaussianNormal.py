import numpy as np

def gaussianmodel(x, mu, sigma):
    coeff_part = 1/(np.sqrt(2*np.pi*sigma**2))
    exp_part = np.exp(-(x-mu)**2 / (2*sigma**2))

    return coeff_part, exp_part

def Model(data):
    mean = np.mean(data)
    stdev = np.std(data)

    pop_model = gaussianmodel(data, mu=mean, sigma=stdev)

    return pop_model

# def LogLikelihood(data, samples):
#     pop_model = Model(data)
#     mu, sigma = pop_model[0], pop_model[1]
#     probabilities = np.zeros(len(samples))
#
#     for n, distance in enumerate(samples):
#         probabilities[n] = gaussianmodel(distance)



