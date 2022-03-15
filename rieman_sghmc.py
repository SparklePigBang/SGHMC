import torch

import pyro
from pyro.ops.integrator import potential_grad

from sghmc import SGHMC

class RiemannSGHMC(SGHMC):
    """Riemann stochastic gradient Hamiltonian Monte Carlo.
    
    Defined in: https://arxiv.org/abs/1506.04696"""
    
    # Update the position one step
    def update_position(self, p, q, potential_fn, step_size):
        return self._step_variable(q, self.step_size, p)

    # Update the momentum one step
    def update_momentum(self, p, q, potential_fn, step_size):
        grad_q, _ = potential_grad(potential_fn, q)
        return self._step_variable(p, - step_size, grad_q)

    # Compute the kinetic energy, given the momentum
    def kinetic_energy(self, p):
        energy = torch.zeros(1)
        for site, value in p.items():
            energy += torch.dot(value, value)
        return 0.5 * energy