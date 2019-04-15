# -*- coding: utf-8 -*-

import numpy as np
import logging
from scipy.stats import boxcox_normmax

#%%---------------------------------------------------------------------------#
class Samples(object):
    
    bias = 1e-10
    
    def __init__(self, *data, **kwargs):
        
        assert data, 'Expect input data.'
        
        if len(data) == 1:
            
            data, = data
            
            assert isinstance(data, np.ndarray) and data.ndim == 2, \
                   'Expect input as 2D-array.'
            
            y = data[:, -1]
            x = data[:, :-1]
            
        else:
            
            y = np.asarray(data[-1]).reshape(-1, 1) 
            
            xs = data[:-1]
            
            if len(xs) == 1:    
                
                x = np.asarray(xs[0]) #np.asarray(xs[0]).reshape(y.size, -1)
                
                if x.ndim == 1:
                    x = np.atleast_2d(x).T
                    
                assert x.ndim == 2 and len(x) == y.size
                
            else:
                
                x = np.vstack(np.asarray(xx).ravel() for xx in xs).T

        assert x.size and y.size == len(x), 'Sizes of x and y are not consistent.'
        
        self.logger = logging.getLogger(__name__)
        
        self.init_data(x.astype(float), y.astype(float), **kwargs)
        
#%%---------------------------------------------------------------------------#
    def init_data(self, x, y, **kwargs):
                        
        self.boxcox = 0
        
        self.size, self.xdim = x.shape
        self.xdeno = np.ones(self.xdim)
        self.xbias = np.zeros(self.xdim)

        self.ymin = 0
        self.ymean = 0
        self.ystd = 1  
        self.yencode = self.ydecode = lambda y: y
        
        if np.unique(y).size > 2:
                           
            self.ymean = y.mean()
            self.ystd = y.std()
            
            self.yencode = lambda y, mean=self.ymean, std=self.ystd: (y - mean) / std
            self.ydecode = lambda y, mean=self.ymean, std=self.ystd: y * std + mean

            if kwargs.get('boxcox', False):
                
                try:
                                
                    y0 = self.yencode(y)
                    self.ymin = y0.min()
                    
                    self.boxcox = boxcox_normmax((y0 - self.ymin + Samples.bias).ravel())
                    
                    assert self.boxcox != 0 
                    
                    self.yencode = (lambda y, boxcox=self.boxcox, ymin=self.ymin,
                                           ymean = self.ymean, ystd=self.ystd,
                                           bias=Samples.bias: 
                                           ((((y - ymean) / ystd) - ymin 
                                             + bias)**boxcox - 1) / boxcox)
                        
                    self.ydecode = (lambda y, boxcox=self.boxcox, ymin=self.ymin,
                                           ymean = self.ymean, ystd=self.ystd,
                                           bias=Samples.bias: 
                                           ((y * boxcox + 1)**(1 / boxcox) 
                                           + ymin - bias) * ystd + ymean)
                    
                except:
                    
                    self.ymin = 0 
                                                                    
                    self.logger.warning('Could not transfer y by using boxcox.')       
        

        xnorm = None if self.xdim == 1 else 'std'
        xnorm = kwargs.get('normalize', xnorm)
        
        self.xnorm = {None: 0, 'ptp': 1, 'std': 2, 'var': 3}[xnorm]
                    
        if xnorm == 'std':
            
            self.xbias = np.mean(x, axis=0)
            
            self.xdeno = np.std(x, axis=0)
        
        elif xnorm == 'var':
            
            self.xbias = np.mean(x, axis=0)
            
            self.xdeno = np.var(x, axis=0)
        
        elif xnorm == 'ptp':
            
            self.xbias = np.min(x, axis=0)
            
            self.xdeno = np.ptp(x, axis=0)            
            
                        
        assert all(self.xdeno != 0), 'Variance of input {0} is zero.'.format(
                                    ', '.join(str(n) for n in np.arange(1, 
                                    self.xdeno.size + 1, dtype=int)[self.xdeno == 0]))
                    
        self.xencode = (lambda x, denominator=self.xdeno, bias=self.xbias: 
                                        (x - bias) / denominator) 
        
        self.xdecode = (lambda x, denominator=self.xdeno, bias=self.xbias:
                                        x * denominator + bias)
               
        self.x = self.xencode(x)        
        self.y = self.yencode(y)
#%%---------------------------------------------------------------------------#
    def todict(self):
        
        return dict(x=self.x[np.newaxis, ...],
                    y=self.y.reshape(-1, 1)[np.newaxis, ...],
                    boxcox=np.array([self.boxcox]).reshape(1, 1, 1),
                    
                    xbias=self.xbias[np.newaxis, np.newaxis, ...],
                    xdeno=self.xdeno[np.newaxis, np.newaxis, ...],
                    xnorm=np.array([self.xnorm], dtype=int).reshape(1, 1, 1),
                    
                    ymin=np.array([self.ymin]).reshape(1, 1, 1),
                    ybias=np.array([Samples.bias]).reshape(1, 1, 1),
                    ymean=np.array([self.ymean]).reshape(1, 1, 1),
                    ystd=np.array([self.ystd]).reshape(1, 1, 1),
                    )
        
#%%---------------------------------------------------------------------------#
if __name__ == '__main__':
    
    x = np.linspace(0, 5, 6) + 1e-9
#    print(x, boxcox_normmax(x), np.arange(100).reshape(10, 10).ndim)
    
    
    s = Samples(np.arange(100).reshape(-1, 5))
    print(s.xnorm)
    print(s.todict())
    
    d = s.todict()
    for key in d.keys():
        
        print(key, d[key].shape)
    
    
    