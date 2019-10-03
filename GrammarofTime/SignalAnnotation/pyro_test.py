import torch
import pyro
import matplotlib.pyplot as plt

pyro.set_rng_seed(101)

def weather():
    cloudy = pyro.distributions.Bernoulli(0.7).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 80.0}[cloudy]
    scale_temp = {'cloudy': 2.0, 'sunny': 5.0}[cloudy]
    temp = pyro.sample("temp", pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()


def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)

for _ in range(10):
    print(geometric(0.2, 5))


# temp = []
# for i in range(100):
#     temp.append(weather()[1])
#
# plt.plot(temp)
# plt.show()
#     # print(weather())