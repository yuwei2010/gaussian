# -*- coding: utf-8 -*-

import numpy as np
import mean
import cov
import lik
import inf
import opt

import itertools as it

from sample import Samples
from tools import mproperty

#%%---------------------------------------------------------------------------#
class GP(object):
    '''
    Base class for GP model.
    '''    
    def __init__(self, *data, **kwargs):
        
        self.samples = Samples(*data, **kwargs)
        
    #%%-----------------------------------------------------------------------#  
    @mproperty
    def posterior(self):
        
        # return (alpha, L, sW)
        return self.get_posterior(nargout=1)
    #%%-----------------------------------------------------------------------#  
    @property
    def hypers(self):
        
        return self.optimizer.get_hypers()
    
    @hypers.setter
    def hypers(self, hypers):
        
        self.optimizer.apply_hypers(np.asarray(hypers).ravel())
        self._posterior = self.get_posterior(nargout=1)        
            
    #%%-----------------------------------------------------------------------#  
    def get_posterior(self, nargout=1):
        
        return self.inf.evaluate(self.mean, self.cov, 
                                 self.lik, self.samples.x, 
                                 self.samples.y, nargout)

    #%%-----------------------------------------------------------------------#  
    def optimize(self, optimizer=None, num_iters=100, threshold=0):
        
        if not isinstance(optimizer, opt.Optimizer):
            optimizer = self.optimizer
        
        old_hypers = self.hypers
        
        optimal_hyp, nlZ, iters = self.optimizer.find_min(num_iters=num_iters)
        
        if iters < threshold:
            
            self.hypers = old_hypers
            
            raise ValueError('Iteration number {0} is lower than the threshold {1}.' 
                             'The results may be wrong.'.format(iters, threshold))
            
        else:
            
            self.hypers = optimal_hyp
            
            return optimal_hyp, nlZ, iters
        
    #%%-----------------------------------------------------------------------#  
    def predict(self, xstars, ys=None, nargout=2):
        
        xs = self.samples.xencode(np.asarray(xstars, 
                                dtype=float).reshape(-1, self.samples.xdim))
        
        if ys is not None:
            ys = self.samples.yencode(np.asarray(ys, dtype=float).reshape(-1, 1))
            
        x = self.samples.x
        
        meanfunc = self.mean
        covfunc = self.cov
        likfunc = self.lik
        posterior = self.posterior
        
        (alpha, L, sW) = posterior
       
        # self-variances
        Kss = covfunc.get_cov(z=xs)   
        
        # cross-covariances
        Ks = covfunc.get_cov(x=x, z=xs)  
        
        ms = meanfunc.get_mean(xs)
            
        # conditional mean fs|f
        fmu = ms + np.dot(Ks.T, alpha)
                
        # L is triangular => use Cholesky parameters (alpha, sW, L)
        if np.all(np.tril(L, -1) == 0):
            
            V = np.linalg.solve(L.T, sW * Ks)
            
            fs2 = Kss - (V * V).sum(axis=0).reshape(-1, 1)

        # L is not triangular => use alternative parametrization
        else:  
            # predictive variances
            fs2 = Kss + (Ks * np.dot(L, Ks)).sum(axis=0).reshape(-1, 1)
        
        # remove numerical noise i.e. negative variances
        fs2 = np.maximum(fs2, 0)  
                
        ymu = fmu
        ystars = self.samples.ydecode(ymu)
                
        if nargout == 1:
            
            return ystars
        
        else:
            
            ys2 = fs2 + np.exp(2. * likfunc.hyp[0])

            ysig = (self.samples.ydecode(ymu + np.sqrt(ys2))
                    - self.samples.ydecode(ymu - np.sqrt(ys2)))
                        
            if nargout == 2:

                return ystars, ysig
            
            else:
                
                lp = likfunc.evaluate(ys, fmu, fs2, None, None, 1)                  
                return ystars, ysig, lp
            
    #%%-----------------------------------------------------------------------#
    def todict(self):
        
        outdict = self.samples.todict()
        
        (alpha, L, sW) = self.posterior
        
        outdict.update(dict(
                            hyp_mean = self.mean.hyp[np.newaxis, np.newaxis, :],
                            hyp_cov = self.cov.hyp[np.newaxis, np.newaxis, :],
                            hyp_lik = self.lik.hyp[np.newaxis, np.newaxis, :],
                            alpha = alpha[np.newaxis, :, :],
                            L = L[np.newaxis, :, :],
                            sW = sW[np.newaxis, :, :],
                            ))

        return outdict     

#%%---------------------------------------------------------------------------#
class GPR(GP):    
    '''
    Model for Gaussian Process Regression
    '''    
    def __init__(self, *data, **kwargs):
        
        super(GPR, self).__init__(*data, **kwargs)
        
        self.mean = kwargs.get('mean', mean.Const(self.samples.y.mean()))
        self.cov = kwargs.get('cov', cov.RBF())
        self.lik = kwargs.get('lik', lik.Gauss())
        self.inf = kwargs.get('inf', inf.Exact())
        
        self.optimizer = kwargs.get('optimizer', opt.Minimize)(self)  
#%%---------------------------------------------------------------------------#

class GPR_FITC(GPR):
    '''
    Model for Gaussian Process Regression FITC
    '''    
    def __init__(self, *data, **kwargs):
        
        super(GPR_FITC, self).__init__(*data, **kwargs)
        
        self.inf = inf.Exact_FITC()
        
        self.induce = kwargs.get('induce', 5)
        
    #%%-----------------------------------------------------------------------#
    @property
    def induce(self):
        
        if isinstance(self._induce, int):
            
            self._induce = np.vstack(it.product(
                            *(np.linspace(arr.min(), arr.max(), self._induce) for
                              arr in self.samples.x.T)))
        
        return self._induce
        
    
    @induce.setter
    def induce(self, value):
        
        assert isinstance(value, (int, np.ndarray))
        
        self._induce = value

    #%%-----------------------------------------------------------------------#
    @property
    def cov(self):
        
        if not isinstance(self._cov, cov.FITC):
            
            self._cov = cov.FITC(self._cov, self.induce)
            
        return self._cov
    
    @cov.setter
    def cov(self, value):
        
        assert isinstance(value, cov.Kernel)
        
        self._cov = value
        
#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    demoData = np.load('regression_data.npz')
    
    x = demoData['x']    # training data
    y = demoData['y']      # training target
    z = demoData['xstar'] # test data


    fun = lambda x : 0.01 * x**3 + x**2 + x * np.exp(x**0.5) + 50 * x * np.sin(0.8 * x**1.5) 
    
    x1 = np.linspace(0, 3 * np.pi, 500).reshape(-1, 1)
    
    y1 = fun(x1).reshape(-1, 1) + np.random.normal(0, 50, x1.size).reshape(-1, 1) 
    
    x0 = np.linspace(0, 3 * np.pi, 4001).reshape(-1, 1)
            
#    x = x1
#    y = y1
#    z = x0
    
    d = 1
    x = (np.tile(x, d))
    z = (np.tile(z, d))
    
    
    print(x.shape)
    
    model = GPR_FITC(x, y)
    
#    print(model.cov.get_cov_der(x=x, der=0))
    
#    model = GPR(x, y)
#    
#    
#    model.cov = cov.Matern(d=7)    
#    model.mean = mean.Linear(d, samples=model.samples)
    

    print(model.hypers)
#    
    print(model.optimize())

        
    ys, ysig, lp = model.predict(z.ravel(), nargout=3)


    alpha, L,  sW = model.posterior

    
    fig = plt.figure()
    
    ax = fig.gca()
    
    ax.plot(x[:, 0], y, '+')
    ax.plot(z[:, 0], ys)

    plt.fill_between(z[:, 0].ravel(), 
                     (ys + ysig).ravel(), 
                     (ys - ysig).ravel(), linewidths=0.0, alpha=0.5)    
    ax.grid(linestyle=':')
    
    plt.show()
    
#    print(model.lik.hyp, model.cov.hyp, model.mean.hyp, model.lik.sn2)
    

    

    


