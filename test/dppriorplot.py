import numpy as np
import matplotlib.pyplot as plt


train_data1 = np.genfromtxt('train1.log')
eta1 = np.genfromtxt('dpmix1-eta.log')
nu1 = np.genfromtxt('dpmix1-nu.log')
ab1 = np.genfromtxt('dpmix1-ab.log')

train_data2 = np.genfromtxt('train2.log')
eta2 = np.genfromtxt('dpmix2-eta.log')
nu2 = np.genfromtxt('dpmix2-nu.log')
ab2 = np.genfromtxt('dpmix2-ab.log')

train_data = np.genfromtxt('train.log')
eta = np.genfromtxt('dpmix-eta.log')
nu = np.genfromtxt('dpmix-nu.log')
ab = np.genfromtxt('dpmix-ab.log')


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

    if (i < eta.shape[0]-1):
        wt = stick*ab[0, i]/(ab[0, i]+ab[1, i])
        stick = stick*ab[1, i]/(ab[0, i]+ab[1, i])
    else:
        wt = stick

    xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')

plt.axes().set_aspect('equal')
plt.title('FullData')


stick = 1.0
D = train_data1.shape[1]
plt.figure()
plt.scatter(train_data1[:, 0], train_data1[:, 1])

for i in range(eta1.shape[0]):
    mu = eta1[i, D*D:D*D+D]/nu1[i]
    psi = np.zeros((D, D))
    for j in range(D):
        psi[j, :] = eta1[i, j*D:(j+1)*D]
    psi -= mu[:, np.newaxis]*mu*nu1[i]
    xi = eta1[i, D*D+D]-D-2
    sig = psi/(xi+D+1)

    if (i < eta1.shape[0]-1):
        wt = stick*ab1[0, i]/(ab1[0, i]+ab1[1, i])
        stick = stick*ab1[1, i]/(ab1[0, i]+ab1[1, i])
    else:
        wt = stick

    xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')

plt.axes().set_aspect('equal')
plt.title('TrData1')


stick = 1.0
D = train_data2.shape[1]
plt.figure()
plt.scatter(train_data2[:, 0], train_data2[:, 1])

for i in range(eta2.shape[0]):
    mu = eta2[i, D*D:D*D+D]/nu2[i]
    psi = np.zeros((D, D))
    for j in range(D):
        psi[j, :] = eta2[i, j*D:(j+1)*D]
    psi -= mu[:, np.newaxis]*mu*nu2[i]
    xi = eta2[i, D*D+D]-D-2
    sig = psi/(xi+D+1)

    if (i < eta2.shape[0]-1):
        wt = stick*ab2[0, i]/(ab2[0, i]+ab2[1, i])
        stick = stick*ab2[1, i]/(ab2[0, i]+ab2[1, i])
    else:
        wt = stick

    xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')

plt.axes().set_aspect('equal')
plt.title('TrData2')

plt.show()

