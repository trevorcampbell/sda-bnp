import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'size' : 24}
matplotlib.rc('font', **font)

model = np.genfromtxt('model-000.log')
train_data = np.genfromtxt('train-000.log')
test_data = np.genfromtxt('test-000.log')
D = int((-1 + np.sqrt(1-4*(1-model.shape[1])))/2)
plt.figure()
plt.scatter(train_data[:, 0], train_data[:, 1], c='b')
plt.scatter(test_data[:, 0], test_data[:, 1], c='g')
for k in range(model.shape[0]):
    mu = model[k, 0:D]
    sig = np.zeros((D, D))
    for j in range(D):
        sig[j, :] = model[k, (j+1)*D:(j+2)*D]
    wt = model[k, -1]
    xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')
plt.axes().set_aspect('equal')
plt.xlim((-100, 100))
plt.ylim((-100, 100))
plt.title('True Data/Model')



#get the name tags
dist_tags = []
for f in os.listdir('.'):
    if f[:6] == 'dist0-' or f[:6] == 'dist1-' or f[:7] == 'dist1r-' or f[:6] == 'dist2-' or f[:6] == 'distf-':
        first_dash = 0
        second_dash = 0
        first_dash = f.find('-')
        second_dash = f[first_dash+1:].find('-')
        dist_tags.append(f[first_dash+1:second_dash+first_dash+1])

dist_tags = sorted(list(set(dist_tags)))

dist_tags = dist_tags[:3]
dtype = ['1r', '2', 'f']# ['0', '1', '1r', '2', 'f']

for i in range(len(dist_tags)):
    for j in range(len(dtype)):
        plt.figure()
        eta = np.genfromtxt('dist'+dtype[j]+'-'+dist_tags[i]+'-eta.log')
        nu = np.genfromtxt('dist'+dtype[j]+'-'+dist_tags[i]+'-nu.log')
        ab = np.genfromtxt('dist'+dtype[j]+'-'+dist_tags[i]+'-ab.log')

        stick = 1.0
        for k in range(eta.shape[0]):
            mu = eta[k, D*D:D*D+D]/nu[k]
            psi = np.zeros((D, D))
            for m in range(D):
                psi[m, :] = eta[k, m*D:(m+1)*D]
            psi -= mu[:, np.newaxis]*mu*nu[k]
            xi = eta[k, D*D+D]-D-2
            sig = psi/(xi+D+1)

            if (k < eta.shape[0]-1):
                wt = stick*ab[0, k]/(ab[0, k]+ab[1, k])
                stick = stick*ab[1, k]/(ab[0, k]+ab[1, k])
            else:
                wt = stick

            xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
            plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')
        plt.title('Dist'+dtype[j]+'-'+dist_tags[i])
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))


plt.show()
