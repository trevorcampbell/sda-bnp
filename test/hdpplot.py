import numpy as np
import matplotlib.pyplot as plt


train_data = np.genfromtxt('train.log')
eta = np.genfromtxt('hdpmix-eta.log')
nu = np.genfromtxt('hdpmix-nu.log')
uv = np.genfromtxt('hdpmix-uv.log')


stick = 1.0
D = train_data.shape[1]

plt.figure()
plt.scatter(train_data[:, 0], train_data[:, 1])

for i in range(eta.shape[0]):
    mu = eta[i, D*D:D*D+D]/nu[i]
    psi = np.zeros((D, D))
    for j in range(D):
        psi[j, :] = eta[i, j*D:(j+1)*D]
    psi -= mu[:, np.newaxis]*mu*nu[i]
    xi = eta[i, D*D+D]-D-2
    sig = psi/(xi+D+1)

    wt = stick*uv[0, i]/(uv[0, i]+uv[1, i])
    stick = stick*uv[1, i]/(uv[0, i]+uv[1, i])


    xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    plt.plot(xy[0, :], xy[1, :], lw=5, alpha=wt)

plt.axes().set_aspect('equal')
plt.show()

