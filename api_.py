# -*- coding: utf-8 -*-

import numpy as np
from functools import wraps

#%%---------------------------------------------------------------------------#
def mproperty(func):
    @property
    @wraps(func)
    def wrapper(owner):
        name = '_' + func.__name__
        
        if not hasattr(owner, name):
            
            setattr(owner, name, func(owner))
            
        return getattr(owner, name)
    
    return wrapper

#%%---------------------------------------------------------------------------#
class API_GP(object):
    
    def __init__(self, **kwargs):
        
        meanfunc = kwargs.pop('mean', None)
        covfunc = kwargs.pop('cov', None)
        
        for key, value in kwargs.items():

            assert isinstance(value, np.ndarray) and value.ndim == 3
            
            setattr(self, key, value)
                
        if not hasattr(self, 'boxcox'):
            self.boxcox = np.array([0.]).reshape(1, 1, 1)
            
    #%%-----------------------------------------------------------------------#   
    @mproperty
    def get_mean(self):
        
        return getattr(API_MEAN(self.hyp_mean), self.mean)
        
                    
    #%%-----------------------------------------------------------------------#   
    @property
    def xdata(self):
        
        return (self.x * self.xdeno + self.xbias).squeeze()

    #%%-----------------------------------------------------------------------#   
    @property
    def ydata(self):
        
        return self.ydecode(self.y).squeeze()
    
    #%%-----------------------------------------------------------------------#   
    def ydecode(self, y):

        if np.all(self.boxcox == 0):
            
            return y * self.ystd + self.ymean

        else:
            
            return ((y * self.boxcox + 1)**(1 / self.boxcox) 
                    + self.ymin - self.ybias) * self.ystd + self.ymean
        
    #%%-----------------------------------------------------------------------#  
    def xencode(self, x):
        
        return (x - self.xbias) / self.xdeno 
    
    #%%-----------------------------------------------------------------------#  
    def zencode(self, z):
        
        xdim = self.x.shape[-1]
        
        if z.ndim == 1 or z.ndim == 2:
            z = z.reshape(1, -1, xdim)
        
        return self.xencode(z)
    
    #%%-----------------------------------------------------------------------#           
    def predict_array(self, z, nargout=1, mean=None, cov=None):
        
        x = self.x
        
        z = self.zencode(z)
        
        ms = self.get_mean(z=z)
        
        ks, kss = self.get_cov(x=x, z=z)
        
        fmu = ms + np.einsum('mij,mik->mjk', ks, self.alpha)
        
        ystars = self.ydecode(fmu).squeeze()
        
        if nargout == 1:
            
            return ystars
        
        V = np.linalg.solve(np.swapaxes(self.L, 1, 2), self.sW * ks)
                
        fs2 = kss - (V * V).sum(axis=1)[:, :, np.newaxis]
                
        sn2 = np.exp(2. * self.hyp_lik)
        
        ys2 = fs2 + sn2
        
        ys2[ys2 < 0] = 0
        
        ysig = (self.ydecode(fmu + np.sqrt(ys2))
              - self.ydecode(fmu - np.sqrt(ys2))).squeeze()

        if nargout == 2:

            return  ystars, ysig    
        
        lp = (-(np.zeros_like(fmu) - fmu)**2 / (sn2 + fs2) / 2. 
                         - np.log(2 * np.pi * (sn2 + fs2)) / 2.) 
        
        return ystars, ysig, lp.squeeze() 
    
    #%%-----------------------------------------------------------------------#    
    def get_mean(self, z):
        
        assert mean is not None 
        
        return getattr(API_MEAN(self.hyp_mean), mean)(z)
        
    #%%-----------------------------------------------------------------------#    
    def get_cov(self, x, z):

        assert cov is not None

                
        
#%%---------------------------------------------------------------------------#
class API_MEAN(object):
    
    def __init__(self, hyp):
        
        self.hyp_mean = hyp

    #%%-----------------------------------------------------------------------#     
    def Const(self, z):
        
        zdim, zlen, _ = z.shape
        
        return self.hyp_mean[:, :, -1][:, :, np.newaxis] * np.ones((1, zlen, 1),)
               
    #%%-----------------------------------------------------------------------#       
    def Linear(self, z):
        
        ms_linear = np.einsum('mij,mkj->mik', z, self.hyp_mean[:, :, :-1]) 
        
        return ms_linear
    
#%%---------------------------------------------------------------------------#
class API_COV(object):
    
    def __init__(self, hyp):
        
        self.hyp_cov = hyp

    #%%-----------------------------------------------------------------------#             
    def BRFard(self, x, z):
        
        zdim, zlen, xdim = z.shape
        
        ell = 1. / np.exp(self.hyp_cov[:, :, :-1])
        sf2 = np.exp(2. * self.hyp_cov[:, :, -1][:, :, np.newaxis])
        
        ell_diag = np.einsum('qmj,jk->qjk', ell, np.eye(xdim))
        
        x_ell = np.einsum('mii,mki->mki', ell_diag, x)
        z_ell = np.einsum('mii,mki->mki', ell_diag, z)

        ks = sf2 * np.exp(-0.5 * (np.einsum('mijk->mij', 
                                 ((x_ell)[:, :, np.newaxis, :]
                                 -(z_ell)[:, np.newaxis, :, :])**2)))

        kss = sf2 * np.exp(-0.5 * np.zeros((zdim, zlen, 1)))        
        
        return ks, kss
                   
#%%---------------------------------------------------------------------------#
class MZ_CR(API_GP):
    '''mean = mean.Zero(), cov = cov.RBF()'''  
    
    def get_mean_cov(self, x, z): 
        
        zdim, zlen, _ = z.shape
        ms = np.zeros((zdim, zlen, 1)) 

        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        sf2 = np.exp(2. * self.hyp_cov[:, :, 1][:, :, np.newaxis])
        
        ks = sf2 * np.exp(-0.5 * (np.einsum('mijk->mij', 
                                 ((x / ell)[:, :, np.newaxis, :]
                                 -(z / ell)[:, np.newaxis, :, :])**2)))
        
        kss = sf2 * np.exp(-0.5 * np.zeros((zdim, zlen, 1)))
        
        return ms, ks, kss
    
#%%---------------------------------------------------------------------------#
class MLC_CR(API_GP):
    '''mean = mean.Linear() + mean.Const(), cov = cov.RBFard()'''          
    
    def get_mean(self, z):

        zdim, zlen, _ = z.shape
        
        ms_linear = np.einsum('mij,mkj->mik', z, self.hyp_mean[:, :, :-1]) 
        ms_const = self.hyp_mean[:, :, -1][:, :, np.newaxis] * np.ones((1, zlen, 1),)

        return ms_linear + ms_const     

    #%%-----------------------------------------------------------------------#  
    def get_cov(self, x, z):
        
        zdim, zlen, xdim = z.shape
        
        ell = 1. / np.exp(self.hyp_cov[:, :, :-1])
        sf2 = np.exp(2. * self.hyp_cov[:, :, -1][:, :, np.newaxis])
        
        ell_diag = np.einsum('qmj,jk->qjk', ell, np.eye(xdim))
        
        x_ell = np.einsum('mii,mki->mki', ell_diag, x)
        z_ell = np.einsum('mii,mki->mki', ell_diag, z)

        ks = sf2 * np.exp(-0.5 * (np.einsum('mijk->mij', 
                                 ((x_ell)[:, :, np.newaxis, :]
                                 -(z_ell)[:, np.newaxis, :, :])**2)))

        kss = sf2 * np.exp(-0.5 * np.zeros((zdim, zlen, 1)))

        return ks, kss
                    
#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    xstar = np.load('xstars2.npy')
    data = dict((key, np.load('testdata2.npz')[key]) for 
                key in np.load('testdata2.npz').keys())      
    
    print(xstar.shape, data.keys())
    
    xstar = np.tile(xstar[None, ...], (1, 1, 1), ) 
    
    model = MLC_CR(**data)
    
    ymu, ysig, lp = (model.predict_array(xstar, 3))

    fig = plt.figure()
    
    ax = fig.gca()  
    
    ax.plot(model.xdata, model.ydata, '+')
    
    ax.plot(xstar.squeeze(), ymu)
    
    plt.fill_between(xstar[0, :, :1].ravel(), 
                    (ymu + ysig).ravel(), 
                    (ymu - ysig).ravel(), linewidths=0.0, alpha=0.5)
    
    plt.show()
