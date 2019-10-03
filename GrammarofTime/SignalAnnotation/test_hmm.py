import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm




model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model1 = hmm.GaussianHMM(n_components=2, covariance_type="full")
remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

#starting probability
model.startprob_ = np.array([0.1, 0.9])
model1.startprob_ = np.array([0.2, 0.8])

#State transition probability
model.transmat_ = np.array([[0, 1],
                            [1, 0]])


model1.transmat_ = np.array([[0.8, 0.2],
                            [0.7, 0.3]])


#Means and covars for each state
model.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))

model1.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
model1.covars_ = np.tile(np.identity(2), (3, 1, 1))

#starting probability
model.startprob_ = np.array([0.1, 0.9])
model1.startprob_ = np.array([0.2, 0.8])

# X - feature matrix for individual observations
# Z - observations
X, Z = model.sample(100)
X1, Z1 = model1.sample(100)

#Means and covars for each state
model.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))

model1.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
model1.covars_ = np.tile(np.identity(2), (3, 1, 1))

#Example - Find probability of a sequence belonging to the model
h_learn = hmm.GaussianHMM(n_components = 2, covariance_type="full",
                                  params="stmc")
h_learn2 = hmm.GaussianHMM(n_components=2, covariance_type="full", params="stmc", algorithm="map")
h_learn.n_iter = 100
h_learn2.n_iter = 100

h_learn.fit(X)
h_learn2.fit(X1)


Z_pred = h_learn.predict(X)
Z_pred2 = h_learn2.predict(X1)