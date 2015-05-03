import numpy as np
import matplotlib.pyplot as plt

d = 0.05
y, x = np.mgrid[slice(-5, 5, d), slice(-5, 5, d)]

data = np.genfromtxt('niwtest.log')

plt.figure()
plt.pcolor(x, y, data)
plt.figure()
plt.plot(x[x.shape[0]/2, :], data[data.shape[0]/2, :])
plt.show()
