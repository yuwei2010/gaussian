# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as spdist

from .sample import Samples
from .tools import mproperty
        
#%%---------------------------------------------------------------------------#
class Kernel(object):
    """
    This is a base class of Kernel functions
    there is no computation in this class, it just defines rules about a kernel class should have
    each covariance function will inherit it and implement its own behaviour
    """

    #%%-----------------------------------------------------------------------#
    @mproperty
    def name(self):
        
        return self.__class__.__name__
    
    #%%-----------------------------------------------------------------------#
    def __add__(self, other):
        
        return Operator_Sum(self, other)  
    
    #%%-----------------------------------------------------------------------#
    def __mul__(self, other):
                    
        return Operator_Product(self, other)
    
    __rmul__ = __mul__ 
    
#%%---------------------------------------------------------------------------#
class FITC(Kernel):
    '''
    Covariance function to be used together with the FITC approximation.
    The function allows for more than one output argument and does not respect the
    interface of a proper covariance function.
    Instead of outputing the full covariance, it returns cross-covariances between
    the inputs x, z and the inducing inputs xu as needed by infFITC
    '''
    def __init__(self, cov, induce):
        
        self.induce = induce
        self.cov = cov
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):
        
        return self.cov.hyp        

    @hyp.setter
    def hyp(self, value):
        
        self.cov.hyp = value
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):

        if x is not None and z is not None:
            # case "cross", compute covariance between data sets x and z
            return self.cov.get_cov(x=self.induce, z=z)

        elif x is not None:
            
            # case "train", compute covariance matix for training set        
            K = self.cov.get_cov(z=x)
            Kuu = self.cov.get_cov(x=self.induce)
            Ku = self.cov.get_cov(x=self.induce, z=x)
        
            return K, Kuu, Ku
        
        elif z is not None:
            # case "self test", # self covariances for the test cases
            return self.cov.get_cov(z=z)
        
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):


        if x is not None and z is not None:
            # case "cross", compute covariance between data sets x and z
            return self.cov.get_cov_der(x=self.induce, z=z, der=der)

        elif x is not None:
            
            # case "train", compute covariance matix for training set        
            K = self.cov.get_cov_der(z=x, der=der)
            Kuu = self.cov.get_cov_der(x=self.induce, der=der)
            Ku = self.cov.get_cov_der(x=self.induce, z=x, der=der)
        
            return K, Kuu, Ku
        
        elif z is not None:
            # case "self test", # self covariances for the test cases
            return self.cov.get_cov_der(z=z, der=der)
   
#%%---------------------------------------------------------------------------#

class Matern(Kernel):    
    '''
    Matern covariance function with nu = d/2 and isotropic distance measure.
    For d=1 the function is also known as the exponential covariance function
    or the Ornstein-Uhlenbeck covariance in 1d.
    d will be rounded to 1, 3, 5 or 7
    hyp = [ log_ell, log_sigma]

    :param d: d is 2 times nu. Can only be 1, 3, 5, or 7
    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    '''

    funcs = {1: lambda A: 1., 
             3: lambda A: 1. + A,
             5: lambda A: 1. + A + A**2 / 3.,
             7: lambda A: 1. + A + 2. * A**2 / 5. + A**3 / 15.,}

    dfuncs = {1: lambda A: 1., 
              3: lambda A: A,
              5: lambda A: (A + A**2) / 3.,
              7: lambda A: (A + 3. * A**2 + A**3) / 15.,}
    
    #%%-----------------------------------------------------------------------#    
    def __init__(self, d=3, log_ell=0., log_sigma=0.):
                    
        self.hyp = np.array([log_ell, log_sigma], dtype=float)
        
        assert d in (1, 3, 5, 7)
        
        self.d = int(d)
        
        self.distfunc = lambda x, ell, d=self.d: np.sqrt(d) * x / ell
        
        self._name = '{0}{1}'.format(self.__class__.__name__, self.d)
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[1])
        
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):

        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = spdist.cdist(self.distfunc(x, self.ell), 
                             self.distfunc(z, self.ell), 'sqeuclidean') 
            
        elif x is not None:
            
            xdist = self.distfunc(x, self.ell)
            # case "train", compute covariance matix for training set
            A = spdist.cdist(xdist, xdist, 'sqeuclidean')
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        A = np.sqrt(self.get_A(x=x, z=z))
        
        return self.sf2 * Matern.funcs[self.d](A) * np.exp(-1. * A) 
        
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
               
       A = np.sqrt(self.get_A(x=x, z=z))
       
       if der == 0:
           
           A = self.sf2 * Matern.dfuncs[self.d](A) * A * np.exp(-1. * A)
       
       elif der == 1:
            
           A = 2 * self.sf2 * Matern.funcs[self.d](A) * np.exp(-1. * A) 
       
       elif der == 2:
               
           A = np.zeros_like(A)
    
       else:
           
           raise ValueError("Wrong derivative value.")
       
       return A
   
#%%---------------------------------------------------------------------------#
class RBF(Matern):
    '''
    Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    '''
    
    def __init__(self, log_ell=0., log_sigma=0.):
        
        super(RBF, self).__init__(d=1, log_ell=log_ell, log_sigma=log_sigma)
        
        self._name = self.__class__.__name__

    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        return  self.sf2 * np.exp(-0.5 * self.get_A(x=x, z=z))
    
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        sf2 = self.sf2
                
        assert der in (0, 1), "Specify the index (0 or 1) of parameters of the derivatives."
        
        A = self.get_A(x=x, z=z)
        
        # compute derivative matrix wrt 1st parameter
        if der == 0:    
            A = sf2 * np.exp(-0.5 * A) * A
            
        # compute derivative matrix wrt 2nd parameter    
        elif der == 1:  
            A = 2. * sf2 * np.exp(-0.5 * A)
            
        return A 

#%%---------------------------------------------------------------------------#
class RBFunit(RBF):
    '''
    Squared Exponential kernel with isotropic distance measure with unit magnitude.
    i.e signal variance is always 1. hyp = [ log_ell ]

    :param log_ell: characteristic length scale.
    '''    
    def __init__(self, log_ell=0.):
        
        super(RBFunit, self).__init__(log_ell=log_ell, log_sigma=0.)   
        
        self.hyp = np.array([log_ell], dtype=float)
        
        self.sf2 = 1.
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])
        
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        assert der == 0, "Wrong derivative index in RDFunit."
        
        return super(RBFunit, self).get_cov_der(x=x, z=z, der=der)

#%%---------------------------------------------------------------------------#       
class RBFard(RBF):
    '''
    Squared Exponential kernel with Automatic Relevance Determination.
    hyp = log_ell_list + [log_sigma]

    :param D: dimension of pattern. set if you want default ell, which is 1 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation.
    '''
    def __init__(self, d=None, log_ell=None, log_sigma=0.):
        
        assert d is not None or log_ell is not None
        
        if isinstance(d, Samples):
            
            d = d.xdim
        
        if log_ell is None:
            
            log_ell = np.zeros(d, dtype=float)
            
        self.hyp = np.r_[np.asarray(log_ell, dtype=float), log_sigma]
        
        self.distfunc = lambda x, ell: np.dot(np.diag(ell), x.T).T
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = 1. / np.exp(self.hyp[:-1])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[-1])   

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        ell = self.ell
                
        A = self.get_cov(x=x, z=z) #sf2 * np.exp(-0.5 * self.get_A(x=x, z=z))
                
        if der == ell.size:
            
            A = 2. * A
            
        elif der < ell.size:
            
            if x is not None and z is not None:
                
                xdist = np.atleast_2d(x[:, der]).T * ell[der]
                zdist = np.atleast_2d(z[:, der]).T * ell[der]
                
                A *= spdist.cdist(xdist, zdist, 'sqeuclidean') 
            
            elif x is not None:
                
                xdist = np.atleast_2d(x[:, der]).T * ell[der]
                
                A *= spdist.cdist(xdist, xdist, 'sqeuclidean')
                
            elif z is not None:
                
                A = np.zeros_like(A) 
            
        else:
            
            raise ValueError("Wrong derivative index in RDFard")
            
        return A
    
#%%---------------------------------------------------------------------------#
class RQ(Kernel):
    '''
    Rational Quadratic covariance function with isotropic distance measure.
    hyp = [ log_ell, log_sigma, log_alpha ]

    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    :param log_alpha: shape parameter for the RQ covariance.
    '''   

    #%%-----------------------------------------------------------------------#    
    def __init__(self, log_ell=0., log_sigma=0., log_alpha=0.):
                    
        self.hyp = np.array([log_ell, log_sigma, log_alpha], dtype=float)
                
        self.distfunc = lambda x, ell: x / ell

    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[1])
        
        # shape parameter for the RQ covariance.
        self.alpha = np.exp(self.hyp[2])
        
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):

        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = spdist.cdist(self.distfunc(x, self.ell), 
                             self.distfunc(z, self.ell), 'sqeuclidean') 
            
        elif x is not None:
            
            xdist = self.distfunc(x, self.ell)
            # case "train", compute covariance matix for training set
            A = spdist.cdist(xdist, xdist, 'sqeuclidean')
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A   

    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        A = self.get_A(x=x, z=z)
        
        return self.sf2 * (1. + 0.5 * A / self.alpha)**(-self.alpha)

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        A = self.get_A(x=x, z=z)
        
        if der == 0:
            
            A = self.sf2 * (1. + 0.5 * A / self.alpha)**(-self.alpha - 1) * A
            
        elif der == 1:
            
            A = 2. * self.sf2 * (1. + 0.5 * A / self.alpha)**(-self.alpha)
            
        elif der == 2:
            
            K = 1. + 0.5 * A / self.alpha
            
            A = self.sf2 * K**(-self.alpha) * (0.5 * A / K - self.alpha * np.log(K))
            
        else:
            
            raise ValueError("Wrong derivative index in covRQ")
            
        return A

#%%---------------------------------------------------------------------------#    
class RQard(RQ):
    '''
    Rational Quadratic covariance function with Automatic Relevance Detemination
    (ARD) distance measure.
    hyp = log_ell_list + [ log_sigma, log_alpha ]

    :param D: dimension of pattern. set if you want default ell, which is 0.5 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation.
    :param log_alpha: shape parameter for the RQ covariance.
    '''
    
    def __init__(self, d=None, log_ell=None, log_sigma=0, log_alpha=0.):
        
        assert d is not None or log_ell is not None
        
        if isinstance(d, Samples):
            
            d = d.xdim
        
        if log_ell is None:
            
            log_ell = np.zeros(d, dtype=float)
            
        self.hyp = np.r_[np.asarray(log_ell, dtype=float), log_sigma, log_alpha]
        
        self.distfunc = lambda x, ell: np.dot(np.diag(ell), x.T).T

    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = 1. / np.exp(self.hyp[:-2])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[-2]) 
        
        # shape parameter for the RQ covariance.
        self.alpha = np.exp(self.hyp[-1])  
        

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        ell = self.ell
                
        A = self.get_A(x=x, z=z) 
        
        if der < ell.size:
            
            if x is not None and z is not None:
                
                xdist = np.atleast_2d(x[:, der]).T / ell[der]
                zdist = np.atleast_2d(z[:, der]).T / ell[der]
                
                A = (self.sf2 * (1. + 0.5 * A / self.alpha)**(-(self.alpha + 1))
                    * spdist.cdist(xdist, zdist, 'sqeuclidean')) 
            
            elif x is not None:
                
                xdist = np.atleast_2d(x[:, der]) / ell[der]
                
                A = (self.sf2 * (1. + 0.5 * A / self.alpha)**(-(self.alpha + 1))
                    * spdist.cdist(xdist, xdist, 'sqeuclidean'))
                
            elif z is not None:
                
                A = np.zeros_like(A) 
        
        elif der == ell.size:
            
            A = 2. * self.sf2 * ( 1. + 0.5 * A / self.alpha )**(-self.alpha) 
            
        elif der == (ell.size + 1):   
            
            K = 1. + 0.5 * A / self.alpha
            A = self.sf2 * K**(-self.alpha) * ( 0.5 * A / K - self.alpha * np.log(K))  
            
        else:
            
            raise ValueError("Wrong derivative index in covRQard")
            
        return A
    
#%%---------------------------------------------------------------------------#
class Const(Kernel):
    '''
    Constant kernel. hyp = [ log_sigma ]

    :param log_sigma: signal deviation.
    '''
    def __init__(self, log_sigma=0.):    
        
        self.hyp = np.array([log_sigma], dtype=float)
    
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # signal variance
        self.sf2 = np.exp(self.hyp[0])  
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):

        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = np.ones((len(x), len(z))) 
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set
            A = np.ones((len(x), len(x)))
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.ones((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        return self.sf2 * self.get_A(x=x, z=z)
    
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
                
        return 2. * self.sf2 *  self.get_A(x=x, z=z)
    
#%%---------------------------------------------------------------------------#

class Linear(Const):
    
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):
        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = np.dot(x, z.T)
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set
            A = np.dot(x, x.T)
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = (z**2).sum(axis=1).reshape(-1, 1)
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A    
    
#%%---------------------------------------------------------------------------# 
class LINard(Kernel):
    '''
    Linear covariance function with Automatic Relevance Detemination.
    hyp = log_ell_list

    :param D: dimension of training data. Set if you want default ell, which is 1 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    '''    
    def __init__(self, d=None, log_ell=None):
        
        assert d is not None or log_ell is not None
        
        if isinstance(d, Samples):
            
            d = d.xdim
        
        if log_ell is None:
            
            log_ell = np.zeros(d, dtype=float)
            
        self.hyp = np.asarray(log_ell, dtype=float)

    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # signal variance
        self.ell = np.exp(self.hyp) 
        
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):

        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = np.dot(x, np.dot(z, np.diag(1. / self.ell)).T)
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set
            A = np.dot(x, x.T)
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = (z**2).sum(axis=1).reshape(-1, 1)
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A  

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            
            z = np.dot(z, np.diag(1. / self.ell))
            A = -2. * np.dot(x[:, der].reshape(-1, 1), 
                             z[:, der].reshape(1, -1))
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set
            
            x = x[:, der].reshape(1, -1)            
            A = -2. * np.dot(x.T, x) 
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            
            z = z[:, der].reshape(-1, 1)
            A = -2. * z**2
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return A  
#%%---------------------------------------------------------------------------#
class Poly(Linear):
    '''
    Polynomial covariance function. hyp = [ log_c, log_sigma ]

    :param log_c: inhomogeneous offset.
    :param log_sigma: signal deviation.
    :param degree: degree of polynomial (not treated as hyperparameter, i.e. will not be trained).
    '''
    def __init__(self, degree=2, log_offset=0., log_sigma=0.):
                    
        self.hyp = np.array([log_offset, log_sigma], dtype=float)
        self.degree = int(degree)
        
        assert self.degree >= 1

    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.offset = np.exp(self.hyp[0])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[1])
        
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        return self.sf2 * (self.offset + self.get_A(x=x, z=z))**self.degree
    
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        A = self.get_A(x=x, z=z)
        
        if der == 0:
            
            A = self.offset * self.degree * self.sf2 * (self.offset + A)**(self.degree - 1)
            
        elif der == 1:
            
            A = 2. * self.sf2 * (self.offset + A)**self.degree
            
        else:
            
            A = np.zeros_like(A)
            
        return A
    
#%%---------------------------------------------------------------------------#
class PiecePoly(Kernel):
    '''
    Piecewise polynomial kernel with compact support.
    hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    :param v: degree v will be rounded to 0,1,2,or 3. (not treated as hyperparameter, i.e. will not be trained).
    '''

        
    #%%-----------------------------------------------------------------------#
    def __init__(self, degree=2, log_ell=0., log_sigma=0. ):

        funcs = {0: lambda A, j: 1., 
                 1: lambda A, j: 1. + (j + 1) * A,
                 2: lambda A, j: 1. + (j + 2) * A + (j**2 + 4 * j + 3) / 3 * A**2,
                 3: lambda A, j: (1. + (j + 3) * A + (6 * j**2 + 36.* j + 45) / 15 * A**2
                                  + (j**3 + 9 * j**2 + 23 * j + 15) / 15 * A**3),}
    
        dfuncs = {0: lambda A, j: 0., 
                  1: lambda A, j: 1. + j,
                  2: lambda A, j: 2. + j + 2 * (j**2 + 4 * j + 3) / 3 * A,
                  3: lambda A, j: (3. + j + 2 * (6 *j**2 + 36 * j + 45) / 15 * A 
                                  + (j**3 + 9 * j**2 + 23 * j + 15) / 5 * A**2),}
        
        ppmax = lambda A: np.maximum(A, np.zeros_like(A))
        
        self.hyp = np.array([log_ell, log_sigma], dtype=float)
        self.degree = int(degree)
        
        assert self.degree in (0, 1, 2, 3)
        
        self.pp = lambda A, j, v=self.degree: funcs[v](A, j) * ppmax(1 - A)**(j + v)
        self.dpp = lambda A, j, v=self.degree: (ppmax(1 - A)**(j + v - 1) 
                                               * A * ((j + v) * funcs[v](A, j) 
                                               - ppmax(1 - A) * dfuncs[v](A, j))) 
        
        self._name = '{0}{1}'.format(self.__class__.__name__, self.degree)
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])
        
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[1])  
        
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):
                
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = np.sqrt(spdist.cdist(x / self.ell, z / self.ell, 'sqeuclidean'))
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set            
            xdist = x / self.ell            
            A = np.sqrt(spdist.cdist(xdist, xdist, 'sqeuclidean'))
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")

        D = z.shape[1] if x is None else x.shape[1]
        j = np.floor(0.5 * D) + self.degree + 1   
         
        return A, j
    
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):

        return self.sf2 * self.pp(*self.get_A(x=x, z=z))        
        
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        if der == 0:
            
            A = self.sf2 * self.dpp(*self.get_A(x=x, z=z))
            
        elif der == 1:
            
            A = 2. * self.sf2 * self.pp(*self.get_A(x=x, z=z))
        
        else:
            
            A, _ = self.get_A(x=x, z=z)
            
            A = np.zeros_like(A)
                    
        return A

#%%---------------------------------------------------------------------------#    
class Gabor(Kernel):
    '''
    Gabor covariance function with length scale ell and period p. The
    covariance function is parameterized as:

    k(x,z) = h( ||x-z|| ) with h(t) = exp(-t^2/(2*ell^2))*cos(2*pi*t/p).

    The hyperparameters are:

    hyp = [log(ell), log(p)]

    Note that SM covariance implements a weighted sum of Gabor covariance functions, but
    using an alternative (spectral) parameterization.

    :param log_ell: characteristic length scale.
    :param log_p: period.
    '''   

    def __init__(self, log_ell=0., log_period=0.):
        
        self.hyp = np.array([log_ell, log_period], dtype=float)

    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])  
        
        # period
        self.period = np.exp(self.hyp[1])   
        
    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):
                
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = spdist.cdist(x / self.ell, z / self.ell, 'sqeuclidean') 
            
        elif x is not None:
            
            # case "train", compute covariance matix for training set
            xdist = x / self.ell
            A = spdist.cdist(xdist, xdist, 'sqeuclidean')
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        dp = 2 * np.pi * np.sqrt(A) * self.ell / self.period

        return A, dp 
     
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        A, dp = self.get_A(x=x, z=z)
        
        return np.exp(-0.5 * A) * np.cos(dp)

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        A, dp = self.get_A(x=x, z=z)   
        A = np.exp(-0.5 * A) * np.cos(dp)
        
        if der == 0: 
        
            A = dp * A
            
        elif der == 1: 
                       
            A = np.tan(dp) * dp * A
            
        else:
            
            raise ValueError("Wrong derivative entry in Gabor")

        return A
    
#%%---------------------------------------------------------------------------#
class Periodic(Kernel):
    '''
    Stationary kernel for a smooth periodic function.
    hyp = [ log_ell, log_p, log_sigma]

    :param log_p: period.
    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    '''  
    def __init__(self, log_ell=0., log_period=0., log_sigma=0.):    
        
        self.hyp = np.array([log_ell, log_period, log_sigma], dtype=float)
    
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        
        # characteristic length scale
        self.ell = np.exp(self.hyp[0])  
        
        # period
        self.period = np.exp(self.hyp[1])        
             
        # signal variance
        self.sf2 = np.exp(2. * self.hyp[2])  

    #%%-----------------------------------------------------------------------#
    def get_A(self, x=None, z=None):
        
        
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            A = spdist.cdist(x, z, 'sqeuclidean') 
            
        elif x is not None:

            # case "train", compute covariance matix for training set
            A = spdist.cdist(x, x, 'sqeuclidean')
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return np.pi / self.period * np.sqrt(A)
    
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        A = self.get_A(x=x, z=z)
        
        A = (np.sin(A) / self.ell)**2
        
        return self.sf2 * np.exp(-2. * A) 

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        A = self.get_A(x=x, z=z)
        
        if der == 0:
            
            A = (np.sin(A) / self.ell)**2
            A = 4. * self.sf2 * np.exp(-2. * A) * A
            
        elif der == 1:
            
            R = np.sin(A) / self.ell
            A = 4. * self.sf2 / self.ell * np.exp(-2. * R**2) * R * np.cos(A) * A
            
        elif der == 2:
            
            A = (np.sin(A) / self.ell)**2            
            A = 2. * self.sf2 * np.exp(-2. * A)
            
        else:
            raise ValueError("Wrong derivative index in covPeriodic")
            
        return A

#%%---------------------------------------------------------------------------#
class Noise(Kernel):
    '''
    Independent covariance function, i.e "white noise", with specified variance.
    Normally NOT used anymore since noise is now added in liklihood.
    hyp = [ log_sigma ]

    :param log_sigma: signal deviation.
    '''
    def __init__(self, log_sigma=0., tol=1e-9):    
        
        self.hyp = np.array([log_sigma], dtype=float)
        self.tol = tol
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
    
        self.sf2 = np.exp(2. * self.hyp[0])          

    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
                
        if x is not None and z is not None:
            
            # case "cross", compute covariance between data sets x and z
            
            M = spdist.cdist(x, z, 'sqeuclidean')
            A = np.zeros_like(M, dtype=float)
            A[M < self.tol] = 1.
            
        elif x is not None:

            # case "train", compute covariance matix for training set
            A = np.eye(len(x))
            
        elif z is not None:
            
            # case "self test", # self covariances for the test cases
            A = np.zeros((len(z), 1))
            
        else:
            
            raise ValueError("Specify at least one:" 
                             "training input (x) or test input (z) or both.")
            
        return self.sf2 * A 

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        if der == 0:
            
            return 2. * self.get_cov(x=x, z=z)
        
        else:
            raise ValueError("Wrong derivative index in covNoise")
    
#%%---------------------------------------------------------------------------#
class Operator_Sum(Kernel):
    
    def __init__(self, this, other):

        if isinstance(other, (int, float)):
            
            other = Const(other)   
            
        self.this = this
        self.other = other
                
        self.hyp = np.r_[this.hyp, other.hyp]
        
        if any(isinstance(cov.name, (list, tuple)) for cov in [this, other]):
            
            if isinstance(this.name, (list, tuple)):
                
                other._name = [other.name]
            else:
                this._name = [this.name]
            
            self._name = list(this.name) + list(other.name)
        else:
            
            self._name = [this.name, other.name]
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):
        
        return self._hyp
    
    @hyp.setter
    def hyp(self, value):
        
        value = np.asarray(value, dtype=float).ravel()
        
        self._hyp = value
        
        assert value.size == self.hyp.size
                
        self.this.hyp = value[:self.this.hyp.size]
        self.other.hyp = value[self.this.hyp.size:]
        
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        return self.this.get_cov(x=x, z=z) + self.other.get_cov(x=x, z=z)

    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        assert der < self.hyp.size
        
        if der < self.this.hyp.size:
            
            return self.this.get_cov_der(x=x, z=z, der=der)
        
        else:
            
            der = der - self.this.hyp.size
            
            return self.other.get_cov_der(x=x, z=z, der=der)

#%%---------------------------------------------------------------------------#
class Operator_Product(Operator_Sum):
    
    #%%-----------------------------------------------------------------------#
    def get_cov(self, x=None, z=None):
        
        return self.this.get_cov(x=x, z=z) * self.other.get_cov(x=x, z=z)
    
    #%%-----------------------------------------------------------------------#
    def get_cov_der(self, x=None, z=None, der=None):
        
        assert der < self.hyp.size
        
        if der < self.this.hyp.size:
            
            return self.this.get_cov_der(x=x, z=z, der=der) * self.other.get_cov(x=x, z=z)
        
        else:
            
            der = der - self.this.hyp.size
            
            return self.other.get_cov_der(x=x, z=z, der=der) * self.this.get_cov(x=x, z=z)
#%%---------------------------------------------------------------------------#
if __name__ == '__main__':
    
    cov = RQard(3)
    
    
    print(cov.hyp, cov.ell)
    
#    print(cov.ell)
