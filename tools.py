# -*- coding: utf-8 -*-
import logging

import numpy as np
import scipy.linalg.lapack as lapack

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

def jitchol(A, attempts=5):
    
    '''   
    JITCHOL Do a Cholesky decomposition with jitter.
    
    Description:
    
    	U = JITCHOL(A, MAXTRIES) attempts a Cholesky decomposition on the
    	given matrix, if matrix isn't positive definite the function gives a
    	warning, adds 'jitter' and tries again. At the first attempt the
    	amount of jitter added is 1e-6 times the mean of the diagonal.
    	Thereafter the amount of jitter is multiplied by 10 each time it is
    	added again. This is continued for a maximum of 10 times.
    	 Returns:
    	  U - the Cholesky decomposition for the matrix.
    	 Arguments:
    	  A - the matrix for which the Cholesky decomposition is required.
    	  MAXTRIES - the maximum number of times that jitter is added before
    	   giving up (default 10).
    
    	[U, JITTER] = JITCHOL(A, MAXTRIES) attempts a Cholesky decomposition
    	on the given matrix, if matrix isn't positive definite the function
    	adds 'jitter' and tries again. Thereafter the amount of jitter is
    	multiplied by 10 each time it is added again. This is continued for
    	a maximum of 10 times.  The amount of jitter added is returned.
    	 Returns:
    	  U - the Cholesky decomposition for the matrix.
    	  JITTER - the amount of jitter that was added to the matrix.
    	 Arguments:
    	  A - the matrix for which the Cholesky decomposition is required.
    	  MAXTRIES - the maximum number of times that jitter is added before
    	   giving up (default 10)

    :param A: the matrixed to be decomposited
    :param int maxtries: number of iterations of adding jitters
    '''

    A = np.asfortranarray(A)
    
    L, info = lapack.dpotrf(A, lower=1)
    
    if info == 0:
        return L
    
    else:
        
        diagA = np.diag(A)
        
        if np.any(diagA <= 0.):
            
            raise np.linalg.LinAlgError("kernel matrix not positive definite: "
                                        "non-positive diagonal elements")
            
        jitter = diagA.mean() * np.finfo(float).tiny #1e-9
        
        while attempts > 0 and np.isfinite(jitter):
            
            #logging.getLogger(__name__).warning('adding jitter of {:.10e} to '
                                                #'diagnol of kernel matrix for '
                                                #'numerical stability'.format(jitter))

            try:                   
                return np.linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)            
            except:                 
                jitter *= 10                
            finally:                 
                attempts -= 1
                
        raise np.linalg.LinAlgError("kernel matrix not positive definite, even with jitter.")
        
#%%---------------------------------------------------------------------------#
def solve_chol(L, B):
    '''
    Solve linear equations from the Cholesky factorization.
    Solve A*X = B for X, where A is square, symmetric, positive definite. The
    input to the function is L the Cholesky decomposition of A and the matrix B.
    Example: X = solve_chol(chol(A),B)

    :param L: low trigular matrix (cholesky decomposition of A)
    :param B: matrix have the same first dimension of L
    :return: X = A \ B
    '''

    assert (L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0]), 'Wrong sizes of matrix arguments in solve_chol.py'

    return np.linalg.solve(L, np.linalg.solve(L.T, B))

#%%---------------------------------------------------------------------------#

def minimize(f, X, args=tuple(), length=None, red=1.0, verbose=False):
    '''
    | This is a function that performs unconstrained
    | gradient based optimization using nonlinear conjugate gradients. 

    | The function is a straightforward Python-translation of Carl Rasmussen's
    | Matlab-function minimize.m:

    % Minimize a differentiable multivariate function. 
    %
    % Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
    %
    % where the starting point is given by "X" (D by 1), and the function named in
    % the string "f", must return a function value and a vector of partial
    % derivatives of f wrt X, the "length" gives the length of the run: if it is
    % positive, it gives the maximum number of line searches, if negative its
    % absolute gives the maximum allowed number of function evaluations. You can
    % (optionally) give "length" a second component, which will indicate the
    % reduction in function value to be expected in the first line-search (defaults
    % to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
    %
    % The function returns when either its length is up, or if no further progress
    % can be made (ie, we are at a (local) minimum, or so close that due to
    % numerical problems, we cannot get any closer). NOTE: If the function
    % terminates within a few iterations, it could be an indication that the
    % function values and derivatives are not consistent (ie, there may be a bug in
    % the implementation of your "f" function). The function returns the found
    % solution "X", a vector of function values "fX" indicating the progress made
    % and "i" the number of iterations (line searches or function evaluations,
    % depending on the sign of "length") used.
    %
    % The Polack-Ribiere flavour of conjugate gradients is used to compute search
    % directions, and a line search using quadratic and cubic polynomial
    % approximations and the Wolfe-Powell stopping criteria is used together with
    % the slope ratio method for guessing initial step sizes. Additionally a bunch
    % of checks are made to make sure that exploration is taking place and that
    % extrapolation will not be unboundedly large.
    %
    % Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).
    '''

    # don't reevaluate within 0.1 of the limit of the current bracket
    INT = 0.1   
    
    # extrapolate maximum 3 times the current step-size        
    EXT = 3 
    
    # max 20 function evaluations per line search              
    MAX = 20  

    # maximum allowed slope ratio               
    RATIO = 10  

    # SIG and RHO are the constants controlling the Wolfe-
    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the
    # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with RHO.               
    SIG = 0.1    
    RHO = SIG / 2.0 
    
    # SMALL = 10.**-16  minimize.m uses matlab's realmin
    SMALL = np.finfo(float).tiny

    # zero the run length counter
    i = 0    
    
    # no previous line search has failed                                      
    ls_failed = 0      
                     
    result = f(X, *args)
    
    # get function value and gradient
    f0 = result[0]  
                           
    df0 = result[1] 
    
    fX = [f0]
    
    # count epochs?!
    i = i + (length < 0)  
                                        
    s = -df0 
    
    # initial search direction (steepest) and slope
    d0 = -np.dot(s, s)  

    # initial step is red/(|s|+1)             
    x3 = red / (1.0 - d0)  
    
    # while not finished
    while i < abs(length):  
        
        # count iterations?!                               
        i = i + (length > 0)                                  
        
        # make a copy of current values
        X0 = X; F0 = f0; dF0 = df0     
          
        if length > 0:
            M = MAX
        else: 
            M = min(MAX, -length - i)  
        
        # keep extrapolating as long as necessary
        while 1:                       
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            
            success = 0 

            while (not success) and (M > 0):
                try:
                    # count epochs?!
                    M = M - 1
                    i = i + (length < 0)   
                    
                    result3 = f(X + x3 * s, *args)
                    
                    f3 = result3[0]
                    df3 = result3[1]
                    
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        return None
                    
                    success = 1
                
                # catch any error which occured in f
                except:
                    # bisect and try again                      
                    x3 = (x2 + x3) / 2.0   

            if f3 < F0:
                # keep best values
                X0 = X + x3*s; F0 = f3; dF0 = df3               
            
            # new slope
            d3 = np.dot(df3, s)       
            
            # are we done extrapolating?
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:                                                      
                break
            
            # move point 2 to point 1
            x1 = x2; f1 = f2; d1 = d2  
            
            # move point 3 to point 2
            x2 = x3; f2 = f3; d2 = d3 
            
            # make cubic extrapolation
            A = 6. * (f1 - f2) + 3. * (d2 + d1) * (x2 - x1)           
            B = 3. * (f2 - f1) - (2. * d1 + d2) * (x2 - x1)
            Z = B + np.sqrt(complex(B * B - A * d1 * (x2 - x1)))
            
            if Z != 0.0:
                # num. error possible, ok!
                x3 = x1 - d1 * (x2 - x1)**2 / Z   
                
            else: 
                x3 = np.inf
            
            # num prob | wrong sign?
            if (not np.isreal(x3)) or np.isnan(x3) or np.isinf(x3) or (x3 < 0): 
                # extrapolate maximum amount                                       
                x3 = x2*EXT
                
            # new point beyond extrapolation limit?           
            elif x3 > x2 * EXT:
                # extrapolate maximum amount
                x3 = x2 * EXT
                
            # new point too close to previous point?             
            elif x3 < x2 + INT * (x2 - x1):   
                x3 = x2 + INT * (x2 - x1)
                
            x3 = np.real(x3) 

        # keep interpolating
        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0: 
            
            # choose subinterval                                                
            if (d3 > 0) or (f3 > f0 + x3 * RHO * d0):  
                # move point 3 to point 4
                x4 = x3; f4 = f3; d4 = d3 
                
            else:
                # move point 3 to point 2
                x2 = x3; f2 = f3; d2 = d3  
                
            if f4 > f0:  
                # quadratic interpolation
                x3 = x2 - ((0.5 * d2 * (x4 - x2)**2) 
                          / (f4 - f2 - d2 * (x4 - x2)))
                                                       
            else:
                # cubic interpolation
                A = 6. * (f2 - f4) / (x4 - x2) + 3. * (d4 + d2)            
                B = 3. * (f4 - f2) - (2. * d2 + d4) * (x4 - x2)
                
                if A != 0:
                    # num. error possible, ok!
                    x3 = x2 + ((np.sqrt(B * B - A * d2 * (x4 - x2)**2) - B) / A)
                                                      
                else:
                    x3 = np.inf
                    
            if np.isnan(x3) or np.isinf(x3):
                # if we had a numerical problem then bisect
                x3 = ((x2 + x4) / 2)   
            
            # don't accept too close
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2  +INT * (x4 - x2))  
                                                        
            result3 = f(X + x3 * s, *args)
            
            f3 = result3[0]
            
            df3 = result3[1]
            
            if f3 < F0:
                # keep best values
                X0 = X + x3 * s; F0 = f3; dF0 = df3
            
            # count epochs?!
            M = M - 1; i = i + (length < 0)  
            
            # new slope                    
            d3 = np.dot(df3, s)                                         
        
        
        # if line search succeeded
        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            
            # update variables
            X = X + x3 * s; f0 = f3; fX.append(f0) 
            
            # Polack-Ribiere CG direction
            s = (np.dot(df3, df3) - np.dot(df0, df3)) / np.dot(df0, df0) * s - df3
            
            # swap derivatives                                       
            df0 = df3  
                                       
            d3 = d0; d0 = np.dot(df0, s)
            
            # new slope must be negative
            if d0 > 0: 
                # otherwise use steepest direction                             
                s = -df0; d0 = -np.dot(s,s)    
            
            # slope ratio but max RATIO
            x3 = x3 * min(RATIO, (d3 / (d0 - SMALL)))
            
            # this line search did not fail
            ls_failed = 0 
                       
        else:
            # restore best point so far
            X = X0; f0 = F0; df0 = dF0 
            
            # line search failed twice in a row or we ran out of time, so we give up
            if ls_failed or (i > abs(length)): 
                break 
            
            # try steepest
            s = -df0; d0 = -np.dot(s, s)  
                            
            x3 = (1. / (1. - d0))
            
            # this line search failed
            ls_failed = 1                              
    
    if verbose:   
        
        logging.getLogger(__name__).info(str(fX))
        
    return X, fX, i 


#%%---------------------------------------------------------------------------#
def SCG(f, x, args=(), niters = 100, gradcheck = False, display = 0, 
        flog = False, pointlog = False, scalelog = False, tolX = 1.0e-8, 
        tolO = 1.0e-8, eval = None):
    
    '''Scaled conjugate gradient optimization. '''
    if display: 
        logging.getLogger(__name__).info('***** starting optimization (SCG) *****')

    nparams = len(x)
    
    eps = 1.0e-4
    
    sigma0 = 1.0e-4
    
    result = f(x, *args)
    
    # Initial function value.
    fold = result[0]             
    fnow = fold
    
    # Increment function evaluation counter.
    funcCount = 1  
    
    # Initial gradient.          
    gradnew = result[1]  
    
    # Increment gradient evaluation counter.
    gradold = gradnew
    
    gradCount = 1 

    # Initial search direction.               
    d = -gradnew

    # Force calculation of directional derivs.                 
    success = 1    

    # nsuccess counts number of successes.              
    nsuccess = 0 

    # Initial scale parameter.                
    beta = 1.0 

    # Lower bound on scale.                  
    betamin = 1.0e-15  

    # Upper bound on scale.          
    betamax = 1.0e50    

    # j counts number of iterations.         
    j = 1  

    # Main optimization loop.
    listF = [fold]
    
    if eval is not None:
        
        evalue, timevalue = eval(x, *args)
        evalList = [evalue]
        time = [timevalue] 

    while (j <= niters):  
        
        # Calculate first and second directional derivatives.
        if (success == 1):
            
            mu = np.dot(d, gradnew)
            
            if (mu >= 0):
                
                d = - gradnew
                mu = np.dot(d, gradnew)
                
            kappa = np.dot(d, d)
            
            if (kappa < eps):
                logging.getLogger(__name__).info("FNEW: " + str(fnow))
                #options(8) = fnow
                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF, j

            sigma = sigma0 / np.sqrt(kappa)
            
            xplus = x + sigma * d
            gplus = f(xplus, *args)[1]
            gradCount += 1
            
            theta = (np.dot(d, (gplus - gradnew))) / sigma   

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        
        if (delta <= 0):
            
            delta = beta * kappa
            beta = beta - theta / kappa

        alpha = -mu / delta 

        # Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *args)[0]
        funcCount += 1
        
        Delta = 2 * (fnew - fold) / (alpha * mu)
        
        if (Delta  >= 0):
            
            success = 1
            nsuccess += 1
            
            x = xnew
            fnow = fnew
            listF.append(fnow)
            
            if eval is not None:
                
                evalue, timevalue = eval(x, *args)
                evalList.append(evalue)
                time.append(timevalue) 
                
        else:
            success = 0
            fnow = fold

        if display > 0:
            logging.getLogger(__name__).info('***** Cycle %4d  Error %11.6f  Scale %e', j, fnow, beta)
            

        if (success == 1):
            
        # Test for termination
        # print type (alpha), type(d), type(tolX), type(fnew), type(fold)
            if ((max(abs(alpha*d)) < tolX) & (abs(fnew-fold) < tolO)):

                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF, j
                
            else:
                
                # Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = f(x, *args)[1]
                gradCount += 1
                
                # If the gradient is zero then we are done.
                if (np.dot(gradnew, gradnew) == 0):
                    
                    # print "FNEW: " , fnew
                    # options(8) = fnew;
                    if eval is not None:
                        return x, listF, evalList, time
                    else:
                        return x, listF, j

        # Adjust beta according to comparison ratio.
        if (Delta < 0.25):
            beta = min(4.0*beta, betamax)
        if (Delta > 0.75):
            beta = max(0.5*beta, betamin)
     
        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if (nsuccess == nparams):
            
            d = -gradnew
            nsuccess = 0
            
        else:
            
            if (success == 1):
                
                gamma = np.dot((gradold - gradnew), gradnew) / mu
                
                d = gamma*d - gradnew

        j += 1
     
    # If we get here, then we haven't terminated in the given number of iterations.
    if (display):
        logging.getLogger(__name__).info("maximum number of iterations reached")
        
    if eval is not None:
        return x, listF, evalList, time
    else:
        return x, listF, j
    
#%%---------------------------------------------------------------------------#
if __name__ == '__main__':
    pass









    