import numpy as np 
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from control import *


class StateSpaceKernel:
    def __init__(self, m=1.0, I=1.0, q=1.0, r=1.0, dt=1.0, q__bounds=(1E-5, 1E2), r__bounds=(1E-5, 1E2)):
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

        M = np.diag([m, m, I])
        A = np.zeros((6, 6))
        A[0:3, 0:3] = np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.diag([1/m, 1/m, 1/I])
        C = np.eye(6)
        D = np.zeros((6, 3))

        Q = q*np.eye(6)
        R = r*np.eye(6)

        sys = ss(A,B,C,D, outputs=['x', 'y', 'theta', 'dx', 'dy', 'dtheta'], inputs=['u1', 'u2', 'u3'], name="Planar Ship Dynamics")
        sys = c2d(sys, dt)

        self.P_inf = dlyap(sys.A, Q)

    @property
    def hyperparameter_q(self):
        return Hyperparameter("q", "numeric", self.q_bounds)

    @property
    def hyperparameter_r(self):
        return Hyperparameter("r", "numeric", self.r_bounds)


    
    def step(self, x, u):
        x_next = np.dot(self.A, x) + np.dot(self.B, u)
        y = np.dot(self.C, x) + np.dot(self.D, u)
        return x_next, y
    
    def diag(self):
        return np.diag(self.A)
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        

        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                dt = abs(xi - yj)
                A_power = np.linalg.matrix_power(self.A, dt)
                cov = self.C @ A_power @ self.P_inf @ self.C.T
                if xi == yj:
                    cov += self.R
                K[i, j] = cov.squeeze()  # If scalar output
        return K
    
        
    def is_stationary(self):
        return True