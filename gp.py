# -*- coding: utf-8 -*-

import numpy as np
from . import mean
from . import cov
from . import lik
from . import inf
from . import opt
import logging
import itertools as it

from .sample import Samples

#%%---------------------------------------------------------------------------#
class GP(object):
    '''
    Base class for GP model.
    '''    
    def __init__(self, *data, **kwargs):
        
        self.samples = Samples(*data, **kwargs)
        self.logger = logging.getLogger(__name__)
        
    #%%-----------------------------------------------------------------------#  
    @property
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
              
    #%%-----------------------------------------------------------------------#  
    def get_posterior(self, nargout=1):
        
        return self.inf.evaluate(self.mean, self.cov, 
                                 self.lik, self.samples.x, 
                                 self.samples.y, nargout)

    #%%-----------------------------------------------------------------------#  
    def optimize(self, restart=100, optimizer=None, num_iters=500, threshold=0, config=None, opt_depth=5):
        
        if not isinstance(optimizer, opt.Optimizer):
            optimizer = self.optimizer
        
        old_hypers = self.hypers
        
        optimal_hyp, nlZ, iters = self.optimizer.find_min(
                                  num_iters=num_iters, num_restart=restart,
                                  config=config, opt_depth=opt_depth)
        
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
            
            fs2 = Kss - (V * V).sum(axis=0, keepdims=True).reshape(-1, 1)

        # L is not triangular => use alternative parametrization
        else:  
            # predictive variances
            fs2 = Kss + (Ks * np.dot(L, Ks)).sum(axis=0, keepdims=True).reshape(-1, 1)
        
        # remove numerical noise i.e. negative variances
        fs2 = np.maximum(fs2, 0, out=fs2)  
                
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
    def todict(self, verbose=True):
        
        outdict = self.samples.todict()
        
        (alpha, L, sW) = self.posterior

        
        if verbose:
            outdict.update(dict(
                                hyp_mean = self.mean.hyp[np.newaxis, np.newaxis, :],
                                hyp_cov = self.cov.hyp[np.newaxis, np.newaxis, :],
                                hyp_lik = self.lik.hyp[np.newaxis, np.newaxis, :],
                                alpha = alpha[np.newaxis, :, :],
                                L = L[np.newaxis, :, :],
                                sW = sW[np.newaxis, :, :],
                                mean = self.mean.name,
                                cov = self.cov.name,
                                ))
        else:
            outdict.update(dict(
                                hyp_mean = self.mean.hyp[np.newaxis, np.newaxis, :],
                                hyp_cov = self.cov.hyp[np.newaxis, np.newaxis, :],
                                hyp_lik = self.lik.hyp[np.newaxis, np.newaxis, :],
                                alpha = alpha[np.newaxis, :, :],
                                mean = self.mean.name,
                                cov = self.cov.name,
                                ))            

        return outdict 
    
    #%%-----------------------------------------------------------------------#    
    def loo(self):
        
        estimator = __import__('api').GP_API(**self.todict(verbose=False))
        
        masks = ~np.eye(estimator.y.size, dtype=bool)
        
        zs = self.samples.xdecode(self.samples.x)[:, np.newaxis, :]
        
        xs = np.vstack(estimator.x[:, mask, :] for mask in masks)
        ys = np.vstack(estimator.y[:, mask, :] for mask in masks)
        
        while 1:
            try:
               
                alphas = np.vstack(self.inf.evaluate(self.mean, self.cov, self.lik,
                         x, y, nargout=1)[0][np.newaxis, :, :] for x, y in zip(xs, ys))
                
                break
            
            except np.linalg.linalg.LinAlgError:
                
                self.lik.hyp = self.lik.hyp * (1 - 1e-3)
                self.logger.warning("Increase lik.hyp to %s for numerical stability", self.lik.hyp)
                
        estimator.x = xs
        estimator.alpha = alphas
       
        return estimator.gpr_predict(zs)
    
    #%%-----------------------------------------------------------------------#  
    def get_r2(self, ystar=None, return_ys=False):
        
        ystar = self.loo() if ystar is None else ystar
        y0 = self.samples.ydecode(self.samples.y).ravel()
        
        r2 = np.corrcoef(y0, ystar)[0, 1]  
        
        if return_ys:            
            return r2, (y0, ystar)        
        else:        
            return r2
        
    #%%-----------------------------------------------------------------------#  
    def reduce_sample(self, num_sample):
        
        if 0 < num_sample < self.samples.size:
            
            y0 = self.samples.ydecode(self.samples.y).ravel()
            
            ystar = self.loo()
            
            mask = np.argsort((ystar - y0)**2)[::-1][:num_sample]
            
            self.samples.x = self.samples.x[mask]
            self.samples.y = self.samples.y[mask]
            
        return self
    
    #%%-----------------------------------------------------------------------#  
    def retrain(self, step=1, target=None, tol=1e-5, attempt=1, opt_args=None):
        
        opt_args = dict() if opt_args is None else opt_args
        
        counter = 0
                
        num_sample = self.samples.size - step
        
        while 1:
            
            assert 0 < num_sample < self.samples.size        
                                        
            r2, (y0, ystar) = self.get_r2( return_ys=True)   
            
            flag = r2 < (1 - tol) if target is None else r2 < min(1, np.abs(target))
            
            mask = np.argsort((ystar - y0)**2)[:num_sample]               
            self.samples.x = self.samples.x[mask]
            self.samples.y = self.samples.y[mask]
            
            self.optimize(**opt_args)
            
            counter += 1
            num_sample -= step
            
            if flag and counter < attempt:
                continue
                    
            else:
                break
                        
        return self
                   
#%%---------------------------------------------------------------------------#
class GPR(GP):    
    '''
    Model for Gaussian Process Regression
    '''    
    def __init__(self, *data, **kwargs):
        
        super(GPR, self).__init__(*data, **kwargs)
        
        m = kwargs.get('mean', mean.Linear(self.samples) + mean.Const(self.samples))
        c = kwargs.get('cov', cov.RBFard(self.samples))
        l = kwargs.get('lik', lik.Gauss())
        i = kwargs.get('inf', inf.Exact())       
        o = kwargs.get('optimizer', opt.Minimize(self))
        
        try:
            self.mean = m if isinstance(m, mean.Mean) else getattr(mean, m)()
        except AssertionError:
            self.mean = getattr(mean, m)(self.samples)
            
        try:
            self.cov = c if isinstance(c, cov.Kernel) else getattr(cov, c)()
        except AssertionError:
            self.cov = getattr(cov, c)(self.samples)
        except AttributeError:
            self.cov = getattr(cov, c[:-1])(int(c[-1]))
            
        self.lik = l if isinstance(l, lik.Likelihood) else getattr(lik, l)()
        self.inf = i if isinstance(i, inf.Inference) else getattr(inf, i)()        
        self.optimizer = o if isinstance(o, opt.Optimizer) else getattr(opt, o)(self)
        
#%%---------------------------------------------------------------------------#
class GPR_FITC(GPR):
    '''
    Model for Gaussian Process Regression FITC
    '''    
    def __init__(self, *data, **kwargs):
        
        super(GPR_FITC, self).__init__(*data, **kwargs)
        
        self.inf = inf.Exact_FITC()
        
        self.induce = kwargs.get('induce_array', kwargs.get('xu', 5))
        
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
    import time
    
    demoData = np.load('regression_data.npz')
    
    x = demoData['x']    # training data
    y = demoData['y']      # training target
    z = demoData['xstar'] # test data

    fun = lambda x : 0.01 * x**3 + x**2 + x * np.exp(x**0.5) + 50 * x * np.sin(0.8 * x**1.5) 
    
    x1 = np.vstack(np.linspace(0, 3 * np.pi, 20).reshape(-1, 1) for i in range(1))
        
    y1 = fun(x1).reshape(-1, 1) #+ np.random.normal(0, 50, x1.size).reshape(-1, 1) 
    
    x0 = np.linspace(0, 3 * np.pi, 4001).reshape(-1, 1)
            
    x = x1
    y = y1
    z = x0
    
    d = 1
    x = (np.tile(x, d))
    z = (np.tile(z, d))
    
    start = time.time()
    
    model = GPR(x, y)
    
    print(model.optimize())
    

  

        
    ys, ysig, lp = model.predict(z, nargout=3)
    
    print(ys.shape)
    fig = plt.figure(tight_layout=True, dpi=120)
    
    ax = fig.gca()
    
#    ax.plot(y, ystars, 'o')
#    
#    plt.show() 
    
    ax.plot(x[:, 0], y, '+', markersize=12)
      
    ax.plot(z[:, 0], ys)
#    ax.plot(x[:, 0], ystars, 'o') 
#    ax.plot(z[:, 0].ravel(), np.poly1d(np.polyfit(x[:, 0].ravel(), y.ravel(), 20))(z[:, 0].ravel()))
#
    plt.fill_between(z[:, 0].ravel(), 
                    (ys + ysig).ravel(), 
                    (ys - ysig).ravel(), linewidths=0.0, alpha=0.3)  
    
    ax.grid(linestyle=':', zorder=0)
    
    plt.show()

    

    

    


