# -*- coding: utf-8 -*-

import numpy as np

#%%---------------------------------------------------------------------------#
class GP_API(object):
    
    def __init__(self, **kwargs):
        
        mean = kwargs.pop('mean', None)
        cov = kwargs.pop('cov', None)
        
        self.mean = mean.tolist() if isinstance(mean, np.ndarray) else mean
        self.cov = cov.tolist() if isinstance(cov, np.ndarray) else cov         
                
        for key, value in kwargs.items():

            assert isinstance(value, np.ndarray) and value.ndim == 3
            
            setattr(self, key, value)
                
        if not hasattr(self, 'boxcox'):
            self.boxcox = np.array([0.]).reshape(1, 1, 1)
            
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
    def predict_array(self, z, nargout=1):
        
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
    def get_mean(self, z, mean=None):
        
        mean = self.mean if mean is None else mean
        
        assert mean is not None 
        
        if isinstance(mean, (list, tuple)):

            start = 0            
            zdim, zlen, _ = z.shape
            res = 0 #np.zeros((zdim, zlen, 1), dtype=float)

            for name in mean:

                ms, lh = getattr(GP_MEAN(self.hyp_mean[:, :, start:]), name)(z)

                res += ms
                start += lh
        else:
            res, _ = getattr(GP_MEAN(self.hyp_mean), mean)(z)
            
        return res
        
    #%%-----------------------------------------------------------------------#    
    def get_cov(self, x, z, cov=None):
        
        cov = self.cov if cov is None else cov
        
        assert cov is not None
        
        if isinstance(cov, (list, tuple)):
            
            start = 0
            _, xlen, _ = x.shape
            zdim, zlen, _ = z.shape
            
            ks = 0 #np.zeros((zdim, xlen, zlen), dtype=float)
            
            kss = 0 #np.zeros((zdim, zlen, 1), dtype=float)
            
            for name in cov:
                
                dks, dkss, lh = getattr(GP_COV(self.hyp_cov[:, :, start:]), name)(x=x, z=z)
                
                ks += dks
                kss += dkss
                start += lh
        else:
            
            if hasattr(cov, '__call__'):               
                ks, kss, _ = cov(x=x, z=z)
            
            else:
                ks, kss, _ = getattr(GP_COV(self.hyp_cov), cov)(x=x, z=z)

        return ks, kss
        
#%%---------------------------------------------------------------------------#
class GP_MEAN(object):
    
    def __init__(self, hyp):
        
        self.hyp_mean = hyp

    #%%-----------------------------------------------------------------------#     
    def Zero(self, z):

        zdim, zlen, _ = z.shape
        ms = np.zeros((zdim, zlen, 1))
        
        return ms, 0
    
    #%%-----------------------------------------------------------------------#     
    def Const(self, z):
        
        zdim, zlen, _ = z.shape
        
        ms = self.hyp_mean[:, :, 0][:, :, np.newaxis] * np.ones((1, zlen, 1),)
        
        return ms, 1
               
    #%%-----------------------------------------------------------------------#       
    def Linear(self, z):
        
        zdim, zlen, xdim = z.shape
        
        ms = np.einsum('mij,mkj->mik', z, self.hyp_mean[:, :, :xdim]) 
        
        return ms, xdim
    
#%%---------------------------------------------------------------------------#
class GP_COV(object):
    
    def __init__(self, hyp):
        
        self.hyp_cov = hyp
        
    #%%-----------------------------------------------------------------------#  
    def Matern(self, x, z, d, fun):
        
        zdim, zlen, _ = z.shape
        
        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        sf2 = np.exp(2. * self.hyp_cov[:, :, 1][:, :, np.newaxis])  
        
        Aks = np.sqrt(np.einsum('mijk->mij', 
            ((np.sqrt(d) * x / ell)[:, :, np.newaxis, :]
            -(np.sqrt(d) * z / ell)[:, np.newaxis, :, :])**2))
        
        Akss = np.zeros((zdim, zlen, 1))
        
        ks = sf2 * fun(Aks) * np.exp(-1. * Aks)
        
        kss = sf2 * fun(Akss) * np.exp(-1. * Akss)
        
        return ks, kss, 2

    #%%-----------------------------------------------------------------------#      
    def Matern1(self, x, z):
        
        fun = lambda A: 1.
        
        return self.Matern(x, z, d=1, fun=fun)

    #%%-----------------------------------------------------------------------#      
    def Matern3(self, x, z):
        
        fun = lambda A: 1. + A
        
        return self.Matern(x, z, d=3, fun=fun)
    
    #%%-----------------------------------------------------------------------#      
    def Matern5(self, x, z):
        
        fun = lambda A: 1. + A + A**2 / 3.
        
        return self.Matern(x, z, d=5, fun=fun)

    #%%-----------------------------------------------------------------------#      
    def Matern7(self, x, z):
        
        fun = lambda A: 1. + A + 2. * A**2 / 5. + A**3 / 15.
        
        return self.Matern(x, z, d=7, fun=fun)  
       
    #%%-----------------------------------------------------------------------#  
    def RBF(self, x, z):
        
        zdim, zlen, _ = z.shape
        
        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        sf2 = np.exp(2. * self.hyp_cov[:, :, 1][:, :, np.newaxis])
        
        ks = sf2 * np.exp(-0.5 * (np.einsum('mijk->mij', 
                                 ((x / ell)[:, :, np.newaxis, :]
                                 -(z / ell)[:, np.newaxis, :, :])**2)))
        
        kss = sf2 * np.exp(-0.5 * np.zeros((zdim, zlen, 1)))
        
        return ks, kss, 2
    
    #%%-----------------------------------------------------------------------#  
    def RBFunit(self, x, z):
        
        zdim, zlen, _ = z.shape
        
        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        
        ks = np.exp(-0.5 * (np.einsum('mijk->mij', 
                           ((x / ell)[:, :, np.newaxis, :]
                         - (z / ell)[:, np.newaxis, :, :])**2)))
        
        kss = np.exp(-0.5 * np.zeros((zdim, zlen, 1)))
        
        return ks, kss, 1
                    
    #%%-----------------------------------------------------------------------#             
    def RBFard(self, x, z):
        
        zdim, zlen, xdim = z.shape
        
        ell = 1. / np.exp(self.hyp_cov[:, :, :xdim])
        sf2 = np.exp(2. * self.hyp_cov[:, :, xdim][:, :, np.newaxis])
        
        ell_diag = np.einsum('qmj,jk->qjk', ell, np.eye(xdim))
        
        x_ell = np.einsum('mii,mki->mki', ell_diag, x)
        z_ell = np.einsum('mii,mki->mki', ell_diag, z)

        ks = sf2 * np.exp(-0.5 * (np.einsum('mijk->mij', 
                                 (x_ell[:, :, np.newaxis, :]
                               -  z_ell[:, np.newaxis, :, :])**2)))

        kss = sf2 * np.exp(-0.5 * np.zeros((zdim, zlen, 1)))        
        
        return ks, kss, xdim + 1

    #%%-----------------------------------------------------------------------#      
    def PiecePoly(self, x, z, d, fun):

        zdim, zlen, xdim = z.shape
        
        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        sf2 = np.exp(2. * self.hyp_cov[:, :, 1][:, :, np.newaxis])
        
        Aks = np.sqrt(np.einsum('mijk->mij', 
                     ((x / ell)[:, :, np.newaxis, :]
                     -(z / ell)[:, np.newaxis, :, :])**2))
        
        Akss = np.zeros((zdim, zlen, 1))
        
        j = np.floor(0.5 * xdim) + d + 1
        
        ks = sf2 * fun(Aks, j)
        kss = sf2 * fun(Akss, j)
        
        return ks, kss, 2

    #%%-----------------------------------------------------------------------#      
    def PiecePoly0(self, x, z):
        
        fun = lambda A, j: np.maximum(1. - A, np.zeros_like(A))**(j + 0)
        
        return self.PiecePoly(x, z, d=0, fun=fun)
    
    #%%-----------------------------------------------------------------------#      
    def PiecePoly1(self, x, z):
        
        fun = lambda A, j: ((1. + (j + 1) * A) 
                            * np.maximum(1. - A, np.zeros_like(A))**(j + 1))
        
        return self.PiecePoly(x, z, d=1, fun=fun)
    
    #%%-----------------------------------------------------------------------#      
    def PiecePoly2(self, x, z):
        
        fun = lambda A, j: ((1. + (j + 2) * A + (j**2 + 4 * j + 3) / 3 * A**2) 
                            * np.maximum(1. - A, np.zeros_like(A))**(j + 2))
        
        return self.PiecePoly(x, z, d=2, fun=fun)
    
    #%%-----------------------------------------------------------------------#      
    def PiecePoly3(self, x, z):
        
        fun = lambda A, j: ((1. + (j + 3) * A + (6 * j**2 + 36.* j + 45.) / 15 * A**2
                             + (j**3 + 9 * j**2 + 23. * j + 15) / 15 * A**3) 
                            * np.maximum(1 - A, np.zeros_like(A))**(j + 3))
        
        return self.PiecePoly(x, z, d=3, fun=fun)
    
    #%%-----------------------------------------------------------------------#        
    def RQ(self, x, z):
        
        zdim, zlen, xdim = z.shape
        
        ell = np.exp(self.hyp_cov[:, :, 0][:, :, np.newaxis])
        sf2 = np.exp(2. * self.hyp_cov[:, :, 1][:, :, np.newaxis])   
        alpha = np.exp(self.hyp_cov[:, :, 2][:, :, np.newaxis])  
        
        ks = sf2 * (1. + 0.5 * np.einsum('mijk->mij', 
                   ((x / ell)[:, :, np.newaxis, :]
                   -(z / ell)[:, np.newaxis, :, :])**2) / alpha)**(-alpha)
        
        kss = sf2 * (1. + 0.5 * np.zeros((zdim, zlen, 1)) / alpha)**(-alpha)

        return ks, kss, 3        
    
    #%%-----------------------------------------------------------------------#        
    def RQard(self, x, z):
        
        zdim, zlen, xdim = z.shape
        
        ell = 1. / np.exp(self.hyp_cov[:, :, :xdim])
        sf2 = np.exp(2. * self.hyp_cov[:, :, xdim][:, :, np.newaxis])   
        alpha = np.exp(self.hyp_cov[:, :, (xdim + 1)][:, :, np.newaxis])  

        ks = sf2 * (1. + 0.5 * np.einsum('mijk->mij', 
                   ((x / ell)[:, :, np.newaxis, :]
                   -(z / ell)[:, np.newaxis, :, :])**2) / alpha)**(-alpha)
        
        kss = sf2 * (1. + 0.5 * np.zeros((zdim, zlen, 1)) / alpha)**(-alpha)

        return ks, kss, xdim + 2 
       
#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    xstar = np.load('xstars2.npy')
    data = dict((key, np.load('testdata2.npz')[key]) for 
                key in np.load('testdata2.npz').keys())      
    
    print(xstar.shape, data.keys(), data['mean'])
    
    xstar = np.tile(xstar[None, ...], (1, 1, 1), ) 


    model = GP_API(**data)
    
    print(model.mean, model.cov)
        
    ymu, ysig, lp = model.gpr_predict(xstar, 3)

    fig = plt.figure()
    
    ax = fig.gca()  
    
    ax.plot(model.xdata, model.ydata, '+')
    
    ax.plot(xstar.squeeze(), ymu)
    
    plt.fill_between(xstar[0, :, :1].ravel(), 
                    (ymu + ysig).ravel(), 
                    (ymu - ysig).ravel(), linewidths=0.0, alpha=0.5)
    
    plt.show()
