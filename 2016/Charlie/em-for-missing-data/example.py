import numpy as np
import matplotlib.pyplot as plt
from mvn import MVN
from matplotlib.patches import Ellipse

# Helper functions
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plotMVN(mu, Sigma, ax, col):
    vals, vecs = eigsorted(Sigma)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    nstd = 3
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=w, height=h,
                  angle=theta, color=col, alpha=0.3)
    ax.add_artist(ell)

# Create Data
dataDim = 2
mu = np.array([1,3])
Sigma = np.array([[1.4, -0.5],[-0.5, 0.8]])
nDataPoints = 200
data = np.random.multivariate_normal(mu, Sigma, nDataPoints)

# Obscure some values MCAR, MAR and NMAR
mcarID = np.random.rand(nDataPoints, 2) > 0.7
marID = data[:,0] > 0.5
nmarID = data[:,0] > 0.5

dataMCAR = data.copy()
dataMCAR[mcarID] = np.nan

dataMAR = data.copy()
dataMAR[marID,1] = np.nan

dataNMAR = data.copy()
dataNMAR[nmarID,0] = np.nan

# Fit MVN to complete data
mvn = MVN(tol=1e-5)
mvn.fit(data)
muComplete = mvn.mu
SigmaComplete = mvn.Sigma

# Fit MVN to MCAR
mvnMCAR = MVN(tol=1e-5)
mvnMCAR.fit(dataMCAR)
muMCAR = mvnMCAR.mu
SigmaMCAR = mvnMCAR.Sigma

# Fit MVN to MAR
mvnMAR = MVN(tol=1e-5)
mvnMAR.fit(dataMAR)
muMAR = mvnMAR.mu
SigmaMAR = mvnMAR.Sigma

# Fit MVN to NMAR
mvnNMAR = MVN(tol=1e-5)
mvnNMAR.fit(dataNMAR)
muNMAR = mvnNMAR.mu
SigmaNMAR = mvnNMAR.Sigma

# Plot and save
ax = plt.subplot(221)
plt.scatter(data[:,0], data[:,1], color='black')
plotMVN(mu, Sigma, plt.subplot(221), 'red')
plt.title('True Parameters')

ax = plt.subplot(222)
plt.scatter(data[:,0], data[:,1], color='black')
plotMVN(muMCAR, SigmaMCAR, plt.subplot(222), 'orange')
plt.title('MCAR Parameters')

ax = plt.subplot(223)
plt.scatter(data[:,0], data[:,1], color='black')
plotMVN(muMAR, SigmaMAR, plt.subplot(223), 'blue')
plt.title('MAR Parameters')

ax = plt.subplot(224)
plt.scatter(data[:,0], data[:,1], color='black')
plotMVN(muNMAR, SigmaNMAR, plt.subplot(224), 'green')
plt.title('NMAR Parameters')

plt.savefig('paramEstimates.pdf')

