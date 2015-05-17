import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(font_scale=1.5)
snsm = sns.color_palette('muted')

nthr_tags = []
for f in os.listdir('.'):
    if f[:19] == 'sdadpmix-adsb-nThr_' and f[-7:] == '-ab.log':
        nthr_tags.append(f[19:-7])
nthr_tags = sorted(list(set(nthr_tags)))

#plot the testll traces first
lghdls = []
lglbls = []

plt.figure()
lglbls = []
lghdls = []
#Plot 6: Test log likelihood vs time with mean/std
for i in range(len(nthr_tags)):
    ntag = nthr_tags[i]
    tr = np.genfromtxt('sdadpmix-adsb-nThr_'+ntag+'-trace.log')
    lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[0], lw=2, alpha=float(i+1)/float(len(nthr_tags)))
    lghdls.append(lghdl)
    lglbls.append('SDA-DP-'+ntag)

tr = np.genfromtxt('vardpmix-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 2], c=snsm[1], lw=2)
lghdls.append(lghdl)
lglbls.append('New Batch')


tr = np.genfromtxt('vardpmixold-adsb-trace.log')
lghdl, = plt.plot(tr[:, 0], tr[:, 1], c=snsm[2], lw=2)
lghdls.append(lghdl)
lglbls.append('Old Batch')

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
plt.legend(lghdls, lglbls)
plt.show()
quit()


#load linear data
data = np.loadtxt('spatial_flows_linear4.log')

#load temporal traces
trace_lat = np.loadtxt('temporal_lats.log')
trace_lon = np.loadtxt('temporal_lons.log')

#load eta, nu, ab
eta = np.loadtxt('eta.log')
nu = np.loadtxt('nu.log')
ab = np.loadtxt('ab.log')

#load US boundaries
bdries = np.loadtxt('boundary_segments.log')

K = eta.shape[0]
N = data.shape[0]
D = data.shape[1]
logpbs = np.zeros((data.shape[0], eta.shape[0]))
#for each datapoint, compute label log probabilities
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
    logpbs[:, k] = np.log(wt) -0.5*np.log(2*np.pi) - 0.5*np.linalg.slogdet(sig)[1] -0.5*np.sum((data-mu)*(np.dot(np.linalg.inv(sig), (data-mu).T).T), axis=1)
    #xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
    #plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')


#get argmax of label probabilities
maxpbs = np.argmax(logpbs, axis=1)

#subsample indices to plot
plot_idcs = np.random.randint(0, N, 1000)

#create seaborn color palette
sns.set(font_scale=1.5)
snsm = sns.color_palette('muted', n_colors=K)

#create the 3d plot
ax = plt.subplot(111, projection='3d')

#plot the united states
for i in range(bdries.shape[0]):
    xs = [np.cos(np.pi/180.0*bdries[i, 1])*np.cos(np.pi/180.0*bdries[i, 0]), np.cos(np.pi/180.0*bdries[i, 3])*np.cos(np.pi/180.0*bdries[i, 2])]
    ys = [np.cos(np.pi/180.0*bdries[i, 1])*np.sin(np.pi/180.0*bdries[i, 0]), np.cos(np.pi/180.0*bdries[i, 3])*np.sin(np.pi/180.0*bdries[i, 2])]
    zs = [np.sin(np.pi/180.0*bdries[i, 1]), np.sin(np.pi/180.0*bdries[i, 3])]
    ax.plot(xs, ys, zs, c='k', lw=2)

#plot each trace on the surface of the unit sphere with color determined by argmax prob
for i in plot_idcs:
    xs = np.cos(np.pi/180.0*trace_lat[i, :])*np.cos(np.pi/180.0*trace_lon[i, :])
    ys = np.cos(np.pi/180.0*trace_lat[i, :])*np.sin(np.pi/180.0*trace_lon[i, :])
    zs = np.sin(np.pi/180.0*trace_lat[i, :])
    ax.plot(xs, ys, zs, c=snsm[maxpbs[i]], lw=0.1)

#display
plt.show()


