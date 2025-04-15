import numpy as np 
import torch
import gpytorch
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter
from control import *

class StateSpaceKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, m=1E5, m_bounds=(1e4, 1e6), 
                 I=1.0, I_bounds=(1E3, 1E6),
                 q=1.0, q_bounds=(1E-5, 1E2),
                 r=1.0, r_bounds=(1E-5, 1E2),
                 dt=1.0, base_kernel=None):
        """
        State space kernel for a planar ship model.
        Parameters:
        m: mass of the ship
        I: moment of inertia of the ship
        q: process noise covariance
        r: measurement noise covariance
        dt: time step
        """
        self.dt = dt
        self.m = m
        self.I = I
        self.q = q
        self.r = r
        
        self.m_bounds = m_bounds
        self.I_bounds = I_bounds
        self.q_bounds = q_bounds
        self.r_bounds = r_bounds

        self.n_states = 6
        self.n_inputs = 3
        self.n_outputs = 6
        
        # Planar ship dynamics
        M = np.diag([m, m, I])
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.diag([1/m, 1/m, 1/I])
        C = np.eye(6)
        D = np.zeros((6, 3))

        self.Q = q*np.eye(6)
        self.R = r*np.eye(6)
        

        self.sys = ss(A,B,C,D, outputs=['x', 'y', 'theta', 'dx', 'dy', 'dtheta'], inputs=['u1', 'u2', 'u3'], name="Planar Ship Dynamics")
        self.sys = c2d(self.sys, dt)
        
        # self.P_inf = dlyap(self.sys.A, self.Q)
        self.base_kernel = base_kernel if base_kernel is not None else RBF(length_scale=1.0)
        
    @property
    def hyperparameter_m(self):
        return Hyperparameter("m", "numeric", self.m_bounds)
    
    @property 
    def hyperparameter_I(self):
        return Hyperparameter("I", "numeric", self.I_bounds)
    
    @property
    def hyperparameter_q(self):
        return Hyperparameter("q", "numeric", self.q_bounds)

    @property
    def hyperparameter_r(self):
        return Hyperparameter("r", "numeric", self.r_bounds)
    
    # def bounds(self):
    #     # Return the bounds of the hyperparameters as a 2D array
    #     return np.array([self.m_bounds, self.I_bounds, self.q_bounds, self.r_bounds])

    # def clone_with_theta(self, theta):
    #     # Clone the kernel with a new set of hyperparameters
    #     return StateSpaceKernel(
    #         m=self.m,
    #         I=self.I,
    #         q=theta[0],
    #         r=theta[1],
    #         dt=self.dt,
    #         q_bounds=self.q_bounds,
    #         r_bounds=self.r_bounds
    #     )
    

    
    # def __call__(self, X, Y=None, eval_gradient=False):
    #     if Y is None:
    #         Y = X
        

    #     for i, xi in enumerate(X):
    #         for j, yj in enumerate(Y):
    #             dt = abs(xi - yj)
    #             A_power = np.linalg.matrix_power(self.A, dt)
    #             cov = self.C @ A_power @ self.P_inf @ self.C.T
    #             if xi == yj:
    #                 cov += self.R
    #             K[i, j] = cov.squeeze()  # If scalar output
    #     return K
    def _latent_force_cov(self, t1, t2):
        cov = np.zeros((self.n_states, self.n_states))

        for k in range(t1):
            Ak = np.linalg.matrix_power(self.A, t1 - 1 - k)
            for l in range(t2):
                Al = np.linalg.matrix_power(self.A, t2 - 1 - l)
                ku = self.input_kernel(self.timesteps[k], self.timesteps[l])
                cov += Ak @ self.B * ku @ self.B.T @ Al.T

        return cov
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        X, Y: 1D arrays of time indices (or 2D if sklearn flattens input)
        """
        X = np.atleast_2d(X).astype(int)
        Y = X if Y is None else np.atleast_2d(Y).astype(int)
        
        K = np.zeros((X.shape[0], Y.shape[0]))

        for i, t1 in enumerate(X[:, 0]):
            for j, t2 in enumerate(Y[:, 0]):
                cov_x = self._latent_force_cov(t1, t2)
                K[i, j] = (self.sys.C @ cov_x @ self.sys.C.T).squeeze()

        if eval_gradient:
            return K, np.zeros((X.shape[0], Y.shape[0], 0))  # No hyperparameter gradients
        return K
        
    # def __call__(self, X, Y=None, eval_gradient=False):
    #     if Y is None:
    #         Y = X
    #     X = np.atleast_1d(X).astype(int).ravel()
    #     Y = np.atleast_1d(Y).astype(int).ravel()
    #     n, m = len(X), len(Y)
    #     K = np.zeros((n, m))
    #     P_inf = self.P_inf

    #     self.A = self.sys.A 
    #     self.B = self.sys.B 
    #     self.C = self.sys.C
    #     # Precompute k_u(t, t′) from the latent force GP
    #     Ku = self.base_kernel(X[:, None], Y[:, None])  # shape: (n, m)

    #     for i, ti in enumerate(X):
    #         Ai = np.linalg.matrix_power(self.A, ti)
    #         CAiB = self.C @ Ai @ self.B  # shape: (output_dim, latent_dim)

    #         for j, tj in enumerate(Y):
    #             Aj = np.linalg.matrix_power(self.A, tj)
    #             CAjB = self.C @ Aj @ self.B

    #             # Latent force contribution
    #             force_cov = Ku[i, j] * (CAiB @ CAjB.T)

    #             # Process noise contribution
    #             d = abs(ti - tj)
    #             Ad = np.linalg.matrix_power(self.A, d)
    #             proc_cov = self.C @ Ad @ P_inf @ Ad.T @ self.C.T

    #             # Total output covariance
    #             cov = force_cov + proc_cov
    #             if ti == tj:
    #                 cov += self.R

    #             K[i, j] = cov.squeeze()  # if 1D output
    #     print(f"K Shape: {K.shape}")
    #     return K
    
    # def diag(self, X):
    #     # return np.diag(self(X))
    #     return np.full(
    #         _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
    #     )

    # def is_stationary(self):
    #     return False
    
    def __repr__(self):
        return "{0}(mass={1:.3g}, inertia={2:.3g})".format(
            self.__class__.__name__, self.m, self.I
        )



class RationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length scales. It is
    parameterized by a length scale parameter :math:`l>0` and a scale
    mixture parameter :math:`\\alpha>0`. Only the isotropic variant
    where length_scale :math:`l` is a scalar is supported at the moment.
    The kernel is given by:

    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}

    where :math:`\\alpha` is the scale mixture parameter, :math:`l` is
    the length scale of the kernel and :math:`d(\\cdot,\\cdot)` is the
    Euclidean distance.
    For advice on how to set the parameters, see e.g. [1]_.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    alpha : float > 0, default=1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'alpha'.
        If set to "fixed", 'alpha' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RationalQuadratic
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8881..., 0.0566..., 0.05518...],
            [0.8678..., 0.0707... , 0.0614...]])
    """

    def __init__(
        self,
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-5, 1e5),
        alpha_bounds=(1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if len(np.atleast_1d(self.length_scale)) > 1:
            raise AttributeError(
                "RationalQuadratic kernel only supports isotropic version, "
                "please use a single scalar for length_scale"
            )
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="sqeuclidean"))
            tmp = dists / (2 * self.alpha * self.length_scale**2)
            base = 1 + tmp
            K = base**-self.alpha
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="sqeuclidean")
            K = (1 + dists / (2 * self.alpha * self.length_scale**2)) ** -self.alpha

        if eval_gradient:
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = dists * K / (self.length_scale**2 * base)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # l is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))

            # gradient with respect to alpha
            if not self.hyperparameter_alpha.fixed:
                alpha_gradient = K * (
                    -self.alpha * np.log(base)
                    + dists / (2 * self.length_scale**2 * base)
                )
                alpha_gradient = alpha_gradient[:, :, np.newaxis]
            else:  # alpha is kept fixed
                alpha_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((alpha_gradient, length_scale_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(alpha={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.alpha, self.length_scale
        )