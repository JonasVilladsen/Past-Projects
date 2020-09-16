# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:41:53 2019

@author: Tobias
"""

import numpy as np
import matplotlib.pyplot as plt

class GCISTA:
    """
    Class for solving problems of the form:
    
        min 0.5*||Ax-y||_2^2 + \lambda*||x||_1 for x in C^n
    
    where:
        A in C^{m\times n} is the measurement matrix (m << n in general)
        y in C^m is the measured data
        x in C^n unknown sparse vector
        lambda > 0 is the reguliser on x (how sparse should solution be)
    
    Functionality:
         - initialise parameters (see __init__)
         - Solve optimization problem using CISTA (see optimize)
         - Analyse convergence of optimization (see optimize agian)
    """
    def __init__(self, A, y, c, var, xmax, T=200, lamb=0.5):
        """
        Initialise parameters CISTA algorithm.
        
        Paramters
        ---------
        A : (m,n) size numpy array
            Mearurement matrix
        y : (m,p) size numpy array
            Measured data. For p > 1 it is a matrix and the solution will be
            a (n,p) size matrix as well
        c : (2K+1) size array
            weights for sum in spline evaluation
        var : positive float
            Variance parameter in basis function
        xmax : positive flaot
            Maximum value expected for input to Ax (element-wise)
        T : int
            Number of iterations in L_ISTA algorithm
        lamb : positive float, optional
            Weight on L_1 reguliser. Default to 0.5
        """
        
        self.A = A
        self.y = y
        self.lamb = lamb
        self.c = c
        self.K = int((np.sqrt(c.shape[0]) - 1)/2)
        self.var = var
        self.xmax = xmax
        self.T = T
        self.m, self.n = A.shape
        
        if len(y.shape) > 1:
            self.p = y.shape[1]
        else:
            self.p = 1
        
        #Run som sanity checks
        assert(isinstance(A, np.ndarray))
        assert(isinstance(y, np.ndarray))
        assert(self.lamb > 0)
        
        #Now calculate some parameters based on A and y
        self.update_parameters(self.A, self.y)
        
        #Grid (k_idx) needs only to be found one time
        self.k_idx = self._get_k_idx(self.n*self.p)
    
    def update_parameters(self, A, y):
        self.m, self.n = A.shape
        self.A = A
        self.y = y
        
        self.A_backward = self._Hermitian(A)
        
        self.alpha = 1/np.linalg.norm(A, ord=2)**2
        
        #Compute matricies involved in gradient step in CISTA
        self.Phi = np.eye(self.n) - self.alpha*self.A_backward.dot(self.A)
            
        self.phi_y = self.alpha*self.A_backward.dot(y)
    
    def optimize(self, x0= 'zeros', x_sol=None, return_mode='sol'):
        """
        Given x0, optimize using CISTA which is a two step optimization 
        algorithm with a gradient step an a proximal step. 
        
        Parameters
        ----------
        x0 : numpy array of size (n,p) or str
            Initial x. defaults to the zero vector/matrix of size (n,p)
        x_sol : numpy array of size (n,p) or None
            If given, the NMSE between x_k and x_sol is tracked at each 
            iteration
        return_mode : str
            all: Return all the above mentioned as d
            sol: Return only sol as float
            obj: return only objective value as float          
            latent: return last x and history of latent variables
            
        Returns
        -------
        Dictionary containing the following:
        xhat : numpy array of size (n,p)
            The found solution x
        obj : float
            Value for the objective function at sol
        history : numpy array
            Objective function for each iteration inclusing for x0 and sol
        """
        if isinstance(x0, str):
            if x0 == 'zeros':
                if self.p == 1:
                    x0 = np.zeros(self.n, dtype="complex")
                else:
                    x0 = np.zeros([self.n, self.p], dtype="complex")
        
        #Run some sanity checks
        else: 
            assert(isinstance(x0, np.ndarray))
            assert(x0.shape[0] == self.n)
            if self.p > 1:
                 assert(x0.shape[1] == self.p)
        assert(return_mode in ('sol', 'latent', 'all'))
        
        #Setup optimization
        x = x0
        
        #If solution is given, both objective cost and NMSE will be tracked
        if x_sol is not None:
            hist = np.zeros([self.T, 2])
            x_MS = np.linalg.norm(x_sol)**2
            hist_update = lambda x: [self._objective(x), self._NMSE(x_sol, x_MS, x)]
        #Else track only objective cost. 
        else:
            hist = np.zeros(self.T)
            hist_update = lambda x: self._objective(x)
        
        #Set initial values
        if self.p > 1:
            z_hist = np.zeros((self.T, self.n, self.p), dtype="complex")
        else:
            z_hist = np.zeros((self.T, self.n), dtype="complex")
            
        #Perform spline_CISTA
        for k in range(self.T):
            
            #Gradient Step
            z = self.Phi.dot(x) + self.phi_y
            
            #Proximal step
            Psi = self._get_Psi(z)
            x = Psi.dot(self.c)
            
            #Reshape back into vector higher dimension
            if self.p > 1:
                x = x.reshape(self.n, self.p, order="F")
            
            z_hist[k] = z
            
            hist[k] = hist_update(x)
        
        if return_mode == 'sol':
            return(x)
        elif return_mode == 'latent':
            return({'xhat': x, 'z_hist': z_hist})
        elif return_mode == 'all':
            return({'xhat': x, 'z_hist': z_hist, 'history': hist})
    
    def _Hermitian(self, X):
        """
        Computes hermitian conjugate of matrix X: conj(X)^T
        """
        return(np.conj(X).T)

    def _objective(self, x):
        """
        Computes the objective
        
        The 2 norm or frobeneous norm will be computed in the first norm 
        depending on if x,y are vectors or matricies. 
        
        x is reshapen in the second norm to get the desired result when 
        x is matrix
        """
        return(0.5*np.linalg.norm(self.A.dot(x) - self.y)**2 +
               self.lamb*np.linalg.norm(x.reshape(self.n*self.p), ord=1))

    def _NMSE(self, x, x_MS, xhat):
        """
        Compute the normalized MSE: ||x-xhat||_2^2/||x||_2^2
        The ||x||_2^2 is given as argument to avoid computing it more than
        once
        """
        return np.linalg.norm(x-xhat)**2/x_MS
    
    def _threshold(self, z):
        """
        Threshold function. This implementation avoids dividing by zero
        accidentially. 
        """
        alpha = self.alpha*self.lamb
        a = np.maximum(np.abs(z), alpha)
        x = (1 - alpha/a)*z
        return(x)
    
    def _mv_normal_0mean_un(self, x):
        """
        Computes multivariate normal, zero mean, unnormed and uncorrelated 
        distribution. 
            exp(-||x||_2^2/2*var)
        
        where the two norm is with respect to the rows of x. If x have more than
        a vector is returned.
        """
        return(np.exp(np.abs(x)**2/(-2*self.var)))
    
    def _matrix_eval(self, Z, func):
        """
        Evaluates complex matrix Z at function that allows vector input.
        Z is reshaped to be a m*n size matrix. Then this array is evaluated at
        given function and finally array is reshaped. 
        """
        m , n = Z.shape
        Z_row = Z.T.reshape((m*n), order="F") #have Z row-Wise
        Z_eval = func(Z_row) 
        return(Z_eval.reshape(m, n))
        
    def _get_k_idx(self,n):
        #Setup values for k_idx for evaluating basis function
        xx = yy = np.linspace(-self.xmax,self.xmax,2*self.K+1)
        xy = np.meshgrid(xx,yy)
        grid = xy[0] + 1j*xy[1]
        grid_vec = grid.reshape((2*self.K+1)**2)
        k_idx = np.tile(grid_vec,n).reshape(n,(2*self.K+1)**2)
        return(k_idx)
    
    def _get_Psi(self, z, k_idx = None, p = None):
        """
        Get Psi matrix as matrix for evaluating product. 
        k_idx should be computed before
        """
        #Use default k_idx if not given (can differ in training) 
        if k_idx is None:
            k_idx = self.k_idx
            
        #Same with p
        if p is None:
            p = self.p
        
        #If z if matrix stack to vector (columns wise)
        if p > 1:
            z = z.T.reshape(self.n*p)
        
        #First reshape z to appropriate matrix Z.
        n = z.size
        Z = np.repeat(z,(2*self.K+1)**2).reshape(n,(2*self.K+1)**2)
        
        #Then create grid
        idx = Z - k_idx
        Psi = self._matrix_eval(idx,func = self._mv_normal_0mean_un)
        return(Psi)
    
    def normalize_rows(self, x):
        """
        function that normalizes each row of the matrix x to have unit length.
    
        Args:
         ``x``: A numpy matrix of shape (n, m)
    
        Returns:
         ``x``: The normalized (by row) numpy matrix.
        """
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

class lrn_GCISTA(GCISTA):
    """
    Class for learning spline coefficients in spl_CISTA
    """
    
    def __init__(self, A, y, n, m, K, xmax, eps=1e-1, overlap=3,
                 extra=1.1, oversample=5, **kwargs):
        """
        Initialse class for learning coefficients c in training example.
        See ?spl_CISTA and ?self.fit_spline_to_threshold for parameters
        """
        #Just initialise c as zero then update
        c = np.zeros((2*K+1)**2); var = 1
        self.n = n
        self.m = m
        
        GCISTA.__init__(self, A, y, c, var, xmax, **kwargs)
        
        #Fit c for K and a
        self.c,self.var = self.fit_spline_to_threshold(xmax,
                                                       extra,
                                                       eps,
                                                       overlap,
                                                       oversample)
        
    def fit_spline_to_threshold(self, xmax, 
                                extra, 
                                eps,
                                overlap,
                                oversample):
        """
        Finds c such that b-spline matches soft threshold function.
        Done by mean square estimator. 
        
        Parameters
        ----------
        xmax : positive float
            What the largest absolute value of x should be fitted for
        extra : positive float larger than one
            How much extra than maximum value of x should be accounted for
            Defaults to 1.1
        eps : positive float
            Basis function only approaches zero. Epsilon determines for which
            value the basis function is deemed zero
        overlap : positive int
            Ratio between diameter of basis function and distance the the farthest
            nearest next basis function on the diagonal direction. 
            Defaults to 2 (ensures Non singular matrix when fitting)
        oversample: positive int
            How many samples pr. parameters. Defaults to 5
        
        Returns
        -------
        
        c : np.ndarray
            Vector containing trained parameters
        var : float
            variance parameter for basis function. This parameter is also
            set in the function. 
        
        """
        #For this optimization p is set to 1

        N = oversample*self.K
        s = xmax*extra/self.K #Distance on diagnoal betwwen two basis functions
        self.var = (overlap*s)**2/(4*np.log(1/eps)) 
        
        #Create grid which is fitted for
        xx = yy = np.linspace(-xmax,xmax,num = N)
        xy = np.meshgrid(xx,yy)
        X = xy[0] + 1j*xy[1]
        x_fit = X.reshape(N**2) #reshape to vector
        
        #update get k_idx for N**2
        k_idx = self._get_k_idx(N**2)
        Psi_fit = self._get_Psi(x_fit, k_idx = k_idx, p = 1) #Design matrix
        y_fit = self._threshold(x_fit) #measurement
        #Fit parameter c
        Phs_fit_H = self._Hermitian(Psi_fit)
        c = np.linalg.inv(Phs_fit_H.dot(Psi_fit)).dot(Phs_fit_H).dot(y_fit)
        return(c , self.var)
        
    def GD(self, N_iter, rho=0.3, mu=1e-4,
           verbose=False,
           generator_function = None,
           generator_parameters = None,
           log_hist = True,
           print_every_percent = 10):
        """
        Find c via gradient descent. 
        
        Parameters
        ----------
        N_iter : positve int
            Number of iterations to find c
        mu : postive float
            Stepsize in gradient decent
        training_generator: python generator object
            Generator object for which the training should be done for
        training_parameters: dict
            Dictionary with training parameters        
        """
        if generator_parameters is None:
            #Sparsity ratio parameter
            generator_parameters = {'rho': rho}
        
        if generator_function is None:
            generator = self.train_generator(**generator_parameters)
            
        else:
            generator = generator_function(**generator_parameters)
        
        #A and y are changed in optimization so save original ones
        A_original = self.A
        y_original = self.y
        
        #For training with matricies reshaping is done such that it is 
        #trained for vectors of shape (n) instead of matricies of 
        #shape n x p. (so p is not considered)
        y = np.ones(self.m)
        A = np.ones((self.m, self.n))
        
        #Not re-initialicze class
        GCISTA.__init__(self, A, y, self.c, self.var, self.xmax,
                        T = self.T, lamb = self.lamb)
        
        #Setup logging of optimization
        if log_hist:
            mse_hist = np.zeros(N_iter)
        
        print_every = int(N_iter/print_every_percent)
        #Run Gradient decent
        for i in range(N_iter):
            if not verbose:
                if i % print_every == 0:
                    print("Iteration {}/{}".format(i+1, N_iter),end = ", ")
            x, A, y = next(generator)
            
            #Update parameters either just y or A and y  
            self.update_parameters(A, y)
            
            x_hat, z_hist = self.optimize(return_mode='latent').values()
            g = self._backprop(z_hist, x_hat, x)
            self.c = self.c - mu*g
            
            if log_hist:
                mse_hist[i] = np.linalg.norm(x_hat-x)**2/(x.size)
                if not verbose:
                    if i % print_every == 0:
                        print("MSE: {:.3e}".format(mse_hist[i]))
            
        #Reset class as original shape
        GCISTA.__init__(self, A_original, y_original, self.c,self.var,
                        self.xmax, self.T, self.lamb)
        if log_hist:
            return({'coef': self.c, 'history': mse_hist})
        else:
            return(self.c)
        
    def _backprop(self, z_hist, x_hat, x_true):
        g0 = np.zeros((2*self.K+1)**2, dtype="complex"); g = g0
        r0 = x_hat - x_true; r = r0
        
        for t in range(self.T-1, -1, -1):
            #Step 1 - no alteration
            z = z_hist[t]
            Psi = self._get_Psi(z)
            

            Psi_grad = self._get_Psi_grad(z)             
            g = g + (Psi.T).dot(r)
            
            phi_grad = Psi_grad.dot(self.c)
            """
            There seems to be two ways to do it:
                1) phi_R*r_R + j*phi_I*r_I
                2) phi*r 
            These are not equivilant but the first option is chosen
            as it corrosponds to the real derivative. 
            """
            #Option 1
            phi_grad_r = phi_grad.real*r.real + 1j*phi_grad.imag*r.imag
            #Option 2
#            phi_grad_r = np.diag(phi_grad).dot(r) #Option 2
            r = self._Hermitian(self.Phi).dot(phi_grad_r)
      
        return(g)
        
    def train_generator(self, rho):
        """
        Generates training example with sparsity rho
        """
        n_sparse = int(np.round(rho*self.n))
        while True:
            x_elements = np.random.normal(0, 1, size=n_sparse) + \
                        1j*np.random.normal(0, 1, size=n_sparse)
            x = np.zeros(self.n, dtype="complex")
            
            x[:n_sparse] = x_elements
            np.random.shuffle(x)
            
            A = np.random.normal(0, 1, size=(self.m, self.n)) + \
                1j*np.random.normal(0, 1, size=(self.m, self.n))
            
            if self.p > 1:
                x = x.reshape(self.n, self.p)
            
            y = A.dot(x)
 
            yield(x, A, y)
            

    def _gradient_normal(self, x):
        """
        Computes the gradient of the 
        "multivariate normal, zero mean, unnormed and uncorrelated" distribution.
        """
        return((-1/self.var)*x*self._mv_normal_0mean_un(x))
        
    def _get_Psi_grad(self, z):
        """
        Get derivative of Psi with respect to z as matrix for evaluating product. 
        k_idx should be computed before
        """
        #Frist reshape z to appropriate matrix Z. (Require z to be vector)
        n = z.size
        Z = np.repeat(z, (2*self.K+1)**2).reshape(n, (2*self.K+1)**2)
        
        #Then create grid
        Psi = self._matrix_eval(Z - self.k_idx, func=self._gradient_normal)
        return(Psi)
        
    def comp_split(self):
        xhat = self.optimize(return_mode='sol')
        l = np.split(xhat, 2)
        x_split = l[0] + 1j*l[1]
        return(x_split)
        
    def plot_learned_threshold(self, x0= None, x1 = None, N=100,
                               plot_difference=False):
        if x0 is None:
            x0 = - 3*self.lamb*self.alpha
        
        if x1 is None:
            x1 =  3*self.lamb*self.alpha
        
        xx = yy = np.linspace(x0,x1,num = N)
        xy = np.meshgrid(xx,yy)
        complex_grid = xy[0] + 1j*xy[1]
        
        x_fit = complex_grid.reshape(N**2)
        k_idx = self._get_k_idx(N**2)
        Psi = self._get_Psi(x_fit, k_idx, p=1)
        y_fit = Psi.dot(self.c)
        Y_fit = y_fit.reshape((N,N))
        
        Y_sh = self._threshold(complex_grid)
        if not plot_difference:
            fig, ax = plt.subplots(1,2)
            ax[0].set_title(r"Threshold")
            ax[0].set_ylabel("Re")
            ax[0].set_xlabel("Im")
            ax[0].imshow(np.abs(Y_sh.T), cmap="plasma",
                          extent=[x0,x1,x0,x1], origin='lower')
            ax[1].set_yticks([])
            ax[1].set_title("Learned Threshold")
            ax[1].set_xlabel("Re")
            img = ax[1].imshow(np.abs(Y_fit.T), cmap="plasma",
                                extent=[x0,x1,x0,x1], origin='lower')
            fig.colorbar(img, ax=ax, orientation='vertical', fraction=.025)
            plt.show()
        else:
            fig, ax = plt.subplots(1,1)
            ax.set_title("Difference between threshold and learned")
            ax.set_ylabel("Re")
            ax.set_xlabel("Im")
            img = ax.imshow(np.abs(Y_sh.T-Y_fit.T),cmap ="plasma",
                            extent=[x0,x1,x0,x1], origin='lower')
            fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.05)
            plt.show()