import os
from scipy.linalg import eigh, inv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#set the palette for the testll plot
sns.set(font_scale=1.5)
snsm = sns.color_palette('muted')

nthr_tags = []
for f in os.listdir('.'):
    if f[:19] == 'sdadpmix-adsb-nThr_' and f[-7:] == '-ab.log':
        nthr_tags.append(f[19:-7])
nthr_tags = sorted(list(set(nthr_tags)))
nthr_tags = [nthr_tags[4]]

#plot the testll traces first
lghdls = []
lglbls = []

plt.figure()
lglbls = []
lghdls = []
#Plot 6: Test log likelihood vs time with mean/std
for i in range(len(nthr_tags)):
    ntag = nthr_tags[i]
    tr = np.genfromtxt('sdadpmix-adsb-nThr_' + ntag + '-trace.log')
    lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[0], lw=2, alpha=float(
        i + 1) / float(len(nthr_tags)))
    lghdls.append(lghdl)
    lglbls.append('SDA-DP')

#tr = np.genfromtxt('vardpmix-adsb-trace.log')
#lghdl, = plt.plot(tr[:, 0], tr[:, 2], c=snsm[1], lw=2)
#lghdls.append(lghdl)
#lglbls.append('New Batch')


tr = np.genfromtxt('vardpmixold-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[2], lw=2)
lghdls.append(lghdl)
lglbls.append('Batch')

tr = np.genfromtxt('svadpmix-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[3], lw=2)
lghdls.append(lghdl)
lglbls.append('SVA')

tr = np.genfromtxt('svidpmix-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[4], lw=2)
lghdls.append(lghdl)
lglbls.append('SVI')

tr = np.genfromtxt('movbdpmix-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[5], lw=2)
lghdls.append(lghdl)
lglbls.append('moVB')

plt.xscale('log')
plt.legend(lghdls, lglbls, loc=2)
plt.xlim([1e-2, 1])
plt.ylim([-5, -.5])
plt.xlabel('Time (s)')
plt.ylabel('Test Log Likelihood')
plt.axes().set_yticklabels(map(float, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...
plt.axes().set_xticklabels(map(float, plt.axes().get_xticks().tolist()))


#load linear data
data = np.loadtxt('linear_flows_3.log')

#load temporal traces
trace_lat = np.loadtxt('spline_lats.log')
trace_lon = np.loadtxt('spline_lons.log')

#load eta, nu, ab
#eta = np.loadtxt('vardpmix-adsb-eta.log')
#nu = np.loadtxt('vardpmix-adsb-nu.log')
#ab = np.loadtxt('vardpmix-adsb-ab.log')

eta = np.loadtxt('sdadpmix-adsb-nThr_016-eta.log')
nu = np.loadtxt('sdadpmix-adsb-nThr_016-nu.log')
ab = np.loadtxt('sdadpmix-adsb-nThr_016-ab.log')


#load US boundaries
bdries = np.loadtxt('boundary_segments.log')

K = eta.shape[0]
K2plot = 10
NpK = 25
N = data.shape[0]
D = data.shape[1]
logpbs = np.zeros((data.shape[0], eta.shape[0]))
#for each datapoint, compute label log probabilities
stick = 1.0
for k in range(eta.shape[0]):
    mu = eta[k, D * D:D * D + D] / nu[k]
    psi = np.zeros((D, D))
    for m in range(D):
        psi[m, :] = eta[k, m * D:(m + 1) * D]
    psi -= mu[:, np.newaxis] * mu * nu[k]
    xi = eta[k, D * D + D] - D - 2
    sig = psi / (xi + D + 1)

    if (k < eta.shape[0] - 1):
        wt = stick * ab[0, k] / (ab[0, k] + ab[1, k])
        stick = stick * ab[1, k] / (ab[0, k] + ab[1, k])
    else:
        wt = stick
    logpbs[:, k] = np.log(wt) - 0.5 * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(sig)[1] - 0.5 * np.sum((data - mu) * (np.dot(np.linalg.inv(sig), (data - mu).T).T), axis=1)
    #xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    #plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')


#get argmax of label probabilities
maxlbls = np.argmax(logpbs, axis=1)
maxprbs = np.amax(logpbs, axis=1)

cts = np.zeros(K)
for k in range(K):
    cts[k] = np.sum(maxlbls == k)
bigKs = np.argsort(cts)[::-1][:K2plot]

#get nonempty clusters
mask = cts[bigKs] > 0
bigKs = bigKs[mask]

#color palette for 3d plot
sns.set(font_scale=1.5)
snsm = sns.color_palette('husl', len(bigKs))

plt.figure()
barlist = plt.bar(np.arange(len(bigKs)), cts[bigKs])
for i in range(len(bigKs)):
    barlist[i].set_color(snsm[i])
plt.ylabel('Count')
plt.xlabel('Cluster')
plt.axes().set_xticks(0.4+np.arange(len(bigKs)))
plt.axes().set_xticklabels(map(int, plt.axes().get_xticks().tolist()))
plt.axes().set_yticklabels(map(int, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...



#create the 3d plot
plt.figure()
ax = plt.subplot(111, projection='3d')
ax._axis3don=False
#ax.view_init(37, -94)
##generate sphere
#spherelats = [ [np.pi/180.0*(-90 + 180.0*lat/59.0) for lat in range(60)] for i in range(120)]
#spherelons = [ [np.pi/180.0*(-180 + 360.0*lon/119.0) for lon in range(120)] for i in range(60)]
##transpose spherelons
#spherelons = [ [row[i] for row in spherelons] for i in range(120)]
#spherex = [ [.90*np.cos(spherelats[i][j])*np.cos(spherelons[i][j]) for j in range(60)] for i in range(120)]
#spherey = [ [.90*np.cos(spherelats[i][j])*np.sin(spherelons[i][j]) for j in range(60)] for i in range(120)]
#spherez = [ [.90*np.sin(spherelats[i][j]) for j in range(60)] for i in range(120)]
#ax.plot_surface(spherex, spherey, spherez, rstride=2, cstride=2, color='w', shade=0)


#plot the united states
for i in range(bdries.shape[0]):
    xs = [np.cos(np.pi / 180.0 * bdries[i, 1]) * np.cos(np.pi / 180.0 * bdries[i, 0]), np.cos(np.pi / 180.0 * bdries[i, 3]) * np.cos(np.pi / 180.0 * bdries[i, 2])]
    ys = [np.cos(np.pi / 180.0 * bdries[i, 1]) * np.sin(np.pi / 180.0 * bdries[i, 0]), np.cos(np.pi / 180.0 * bdries[i, 3]) * np.sin(np.pi / 180.0 * bdries[i, 2])]
    zs = [np.sin(
        np.pi / 180.0 * bdries[i, 1]), np.sin(np.pi / 180.0 * bdries[i, 3])]
    ax.plot(xs, ys, zs, c='k', lw=2)

#M = np.array([[1.0, 0, 1.0/2.0, 0],[0, 1.0, 0, 1.0/2.0], [1.0/2.0, 0, 1.0/3.0, 0], [0, 1.0/2.0, 0, 1.0/3.0]])
#W, V = eigh(M)
#A = inv(np.dot(np.diag(np.sqrt(W)), V.T))
itr = 0
for k in bigKs:
    #get the indices with minimum dist for cluster k
    indsk = np.arange(N)[maxlbls == k]
    prbsk = maxprbs[maxlbls == k]
    maxindsk = (np.argsort(prbsk)[::-1])[:NpK]
    #plot each trace on the surface of the unit sphere with color determined by argmax prob
    for i in maxindsk:
        #plot the trace
        xs = np.cos(np.pi / 180.0 * trace_lat[indsk[i], :]) * \
            np.cos(np.pi / 180.0 * trace_lon[indsk[i], :])
        ys = np.cos(np.pi / 180.0 * trace_lat[indsk[i], :]) * \
            np.sin(np.pi / 180.0 * trace_lon[indsk[i], :])
        zs = np.sin(np.pi / 180.0 * trace_lat[indsk[i], :])
        ax.plot(xs[::5], ys[::5], zs[::5], c=snsm[itr], lw=.2)
        ## plot the line
        #lc = np.dot(A, data[indsk[i], :])
        #xm1 = np.cos(np.pi/180.0*lc[0])*np.cos(np.pi/180.0*lc[1])
        #ym1 = np.cos(np.pi/180.0*lc[0])*np.sin(np.pi/180.0*lc[1])
        #zm1 = np.sin(np.pi/180.0*lc[0])
        #xm2 = np.cos(np.pi/180.0*(lc[0]+lc[2]))*np.cos(np.pi/180.0*(lc[1]+lc[3]))
        #ym2 = np.cos(np.pi/180.0*(lc[0]+lc[2]))*np.sin(np.pi/180.0*(lc[1]+lc[3]))
        #zm2 = np.sin(np.pi/180.0*(lc[0]+lc[2]))
        #ax.plot([xm1, xm2], [ym1, ym2], [zm1, zm2], lw=1, c=snsm[itr])

    ## plot the mean
    #mc = np.dot(A, means[k, :])
    #xm1 = np.cos(np.pi/180.0*mc[0])*np.cos(np.pi/180.0*mc[1])
    #ym1 = np.cos(np.pi/180.0*mc[0])*np.sin(np.pi/180.0*mc[1])
    #zm1 = np.sin(np.pi/180.0*mc[0])
    #xm2 = np.cos(np.pi/180.0*(mc[0]+mc[2]))*np.cos(np.pi/180.0*(mc[1]+mc[3]))
    #ym2 = np.cos(np.pi/180.0*(mc[0]+mc[2]))*np.sin(np.pi/180.0*(mc[1]+mc[3]))
    #zm2 = np.sin(np.pi/180.0*(mc[0]+mc[2]))
    #ax.plot([xm1, xm2], [ym1, ym2], [zm1, zm2], lw=4, c='k')
    itr += 1

plt.figure()
#plot the united states
for i in range(bdries.shape[0]):
    plt.plot([bdries[i, 0], bdries[i, 2]], [bdries[i, 1], bdries[i, 3]], lw=2, c='k')

itr = 0
for k in bigKs:
    # plot the mean
    mu = eta[k, D * D:D * D + D] / nu[k]
    mu[0:2] /= np.sqrt(np.sum(mu[0:2]**2))
    a = -mu[0]/mu[1]
    b = mu[0]*37.0/mu[1]-mu[2]/mu[1]-96.5
    lats = np.linspace(22, 52, 100)
    lons = a*lats+b
    plt.plot([lons[0], lons[-1]],[lats[0], lats[-1]],  lw=2, c=snsm[itr])
    itr += 1
plt.xlim([-128, -65])
plt.ylim([22, 52])

plt.figure()
ax = plt.subplot(111, projection='3d')
mus = eta[bigKs, D*D:D*D+D]/nu[bigKs, np.newaxis]
mus[:, 0:2] /= np.sqrt(np.sum(mus[:, 0:2]**2, axis=1))[:, np.newaxis]
ax.scatter(mus[:, 0], mus[:, 1], mus[:, 2], s=100, c='k')
idcs = np.arange(N)
np.random.shuffle(idcs)
idcs = idcs[:1000]
ax.scatter(data[idcs, 0], data[idcs, 1], data[idcs, 2], c='b')


#display
plt.show()
