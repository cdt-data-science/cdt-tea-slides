import numpy as np

class MVN:
    """Multivariate Gaussian Distribution"""

    def __init__(self, tol=1e-2, maxIter=1000, verbose=True):
        self.tol = tol
        self.maxIter = maxIter
        self.isFitted = False
        self.verbose = verbose

    def _eStep(self, X, params):
        """ E-Step of the EM-algorithm."""

        # Get current params
        Sigma = params['Sigma']
        mu = params['mu']

        observedList = [np.array(np.where(~np.isnan(row))).flatten() for row in X]
        nExamples, dataDim = np.shape(X)

        xTotCum = np.zeros(dataDim)
        xxOuterCum = np.zeros([dataDim, dataDim])
        cumEnt = 0
        nHiddenCum = 0

        # Loop over data points
        for n in range(nExamples):

            # Get missing and visible points
            xo = observedList[n]
            xm = np.setdiff1d(np.arange(dataDim), xo)
            nMiss = len(xm)
            muObs = mu[xo]
            muMiss = mu[xm]
            SigmaObs = Sigma[np.ix_(xo,xo)]
            SigmaMiss = Sigma[np.ix_(xm,xm)]
            SigmaObsMiss = Sigma[np.ix_(xo,xm)]
            SigmaMissObs = Sigma[np.ix_(xm,xo)]
            row = X[n,:]
            rowObserved = row[xo]

            # Simplify for case with no missing data
            if nMiss == 0:
                xTotCum = xTotCum + rowObserved
                xxOuterCum = xxOuterCum + np.outer(rowObserved, rowObserved)

            # Otherwise deal with missing data
            else:
                # Get conditional distribution p(x_miss | x_vis, params)
                meanCond = (muMiss +
                            SigmaMissObs.dot(np.linalg.inv(SigmaObs)).dot(rowObserved - muObs))
                SigmaCond = SigmaMiss - SigmaMissObs.dot(np.linalg.inv(SigmaObs)).dot(SigmaObsMiss)

                # Get sufficient statistics
                xTot = np.empty(dataDim)
                xTot[xo] = rowObserved
                xTot[xm] = meanCond
                xTotCum = xTotCum + xTot

                xxOuter = np.empty([dataDim, dataDim])
                xxOuter[np.ix_(xo, xo)] = np.outer(rowObserved, rowObserved)
                xxOuter[np.ix_(xo, xm)] = np.outer(rowObserved, meanCond)
                xxOuter[np.ix_(xm, xo)] = np.outer(meanCond, rowObserved)
                xxOuter[np.ix_(xm, xm)] = np.outer(meanCond, meanCond) + SigmaCond
                xxOuterCum = xxOuterCum + xxOuter

                # Non constant terms of entopy of p(x_miss| x_obs, params) for
                # computation of log likelihood
                cumEnt = cumEnt + 0.5*np.log(np.linalg.det(SigmaCond))

            # Increment cumulative number of missing vars
            nHiddenCum = nHiddenCum + nMiss

        # Constant entropy term p(z | x_obs, theta)
        constEnt = 0.5*nHiddenCum*(1 + np.log(2*np.pi))

        # Expected complete data log-likelihood
        ell = (
             - 0.5*nExamples*np.log(np.linalg.det(2*np.pi*Sigma))
             - 0.5*np.trace(np.linalg.inv(Sigma).dot(
               xxOuterCum + nExamples*np.outer(mu, mu) - 2*np.outer(mu, xTotCum)))
              )

        # Compute likelihood
        ll = cumEnt + constEnt + ell

        # Store sufficient statistics in dictionary
        ss = {
            'xTot' : xTotCum,
            'xxOuter' : xxOuterCum,
            'nExamples' : nExamples
             }

        return ss, ll

    def _mStep(self, ss):
        """ M-Step of the EM-algorithm.

        The M-step takes the sufficient statistics computed in the E-step, and
        maximizes the expected complete data log-likelihood with respect to the
        parameters.

        Args
        ----
        ss : dict

        Returns
        -------
        params : dict

        """
        mu = 1/ss['nExamples'] * ss['xTot']
        Sigma = 1/ss['nExamples'] * (ss['xxOuter']) - np.outer(mu, mu)

        # Store params in dictionary
        params = {
            'mu' : mu,
            'Sigma' : Sigma,
             }

        return params


    def fit(self, X, paramsInit=None):
        """ Fit the model using EM with data X.

        Args
        ----
        X : array, [nExamples, nFeatures]
            Matrix of training data, where nExamples is the number of
            examples and nFeatures is the number of features.
        """
        nExamples, dataDim = np.shape(X)

        if paramsInit is None:
            params = {
                      'mu' : np.random.normal(size=dataDim),
                      'Sigma' : np.eye(dataDim)
                     }
        else:
            params = paramsInit

        oldL = -np.inf

        for i in range(self.maxIter):

            # E-Step
            ss, ll = self._eStep(X, params)

            # Evaluate likelihood
            if self.verbose:
                print("Iter {:d}   NLL: {:.3f}   Change: {:.3f}".format(i,
                      -ll, -(ll-oldL)), flush=True)

            # Break if change in likelihood is small
            if np.abs(ll - oldL) < self.tol:
                break
            oldL = ll

            # M-step
            params = self._mStep(ss)

        else:
            if self.verbose:
                print("MVN did not converge within the specified tolerance." +
                      " You might want to increase the number of iterations.")

        # Update Object attributes
        self.mu = params['mu']
        self.Sigma = params['Sigma']
        self.trainNll = ll
        self.isFitted = True
        self.dataDim = dataDim

    def sample(self, nSamples=1, noisy=True):
        """Sample from fitted model."""

        if  not self.isFitted:
            print("Model is not yet fitted. First use fit to learn the model"
                   + " params.")
        else:
            dataSamples = np.random.multivariate_normal(self.mu, self.Sigma, nSamples)
            return dataSamples


    def score(self, X):
        """Compute the log-likelihood of each sample """

        if not self.isFitted:
            print("Model is not yet fitted. First use fit to learn the model"
                   + " params.")
        else:
            # Get fitted parameters
            params = {
                     'mu': self.mu,
                     'Sigma' : self.Sigma
                     }

            # Apply one step of E-step to get the total log likelihood
            L = self._eStep(X, params)[1]

            # Divide by number of examples to get average log likelihood
            nExamples = np.shape(X)[0]
            meanLl = L / nExamples
            return meanLl
