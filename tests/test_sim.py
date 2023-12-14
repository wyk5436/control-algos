import unittest
import sys
import numpy as np

sys.path.append('..')
from src.sim import simulate_linear_disc, simulate_nonlinear

class TestMyModule(unittest.TestCase):
    
    def test_linear(self):
        A = np.array([[-0.8,0],[0,0.5]])
        B = np.array([[1],[1]])
        C = np.eye(2)
        u = np.zeros([1,100])
        x0 = np.array([[2],[1]])
        result = simulate_linear_disc(A,B,C,u,x0)
        assert np.linalg.norm(result[:, -1]) < 10**(-5), "The state should go to zero"
    
    def test_nonlinear(self):
        A = np.array([[-0.8,0],[0,0.5]])
        C = np.eye(2)
        x0 = np.array([[2],[1]])
        
        def f(x):
            return A @ x;
        
        def u(k, xk):
            return np.zeros([2,1])
        
        result = simulate_nonlinear(f, u, C, x0, 100)
        assert np.linalg.norm(result[:, -1]) < 10**(-5), "The state should go to zero"

