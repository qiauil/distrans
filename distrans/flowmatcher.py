from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .odesolve import odeint, FixedStepConfig, AdaptiveStepConfig
from .odesolve.solver import ODESolver
from typing import Union

class ConditionalFlowMatcher(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self._unifrom_sampler = torch.distributions.uniform.Uniform(0.,1.)
    
    @abstractmethod
    def psi_t(self,x_0:torch.Tensor,x_1:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        """ push forward function that maps samples $x_0$ from $p_0$ to $x_t$ """
    
    @abstractmethod
    def velocity(self,x_0:torch.Tensor,x_1:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        """ velocity field of the flow """
        
    def sample_t(self,x_1:torch.tensor) -> torch.tensor:
        """
        Sample time index from uniform distribution

        Args:
            x_1 (torch.tensor):Samples from the target distribution. 
                The shape of the returned tensor is [x_1.shape[0],1]

        Returns:
            torch.tensor: Samples of time index
        """
        return self._unifrom_sampler.sample([x_1.shape[0],1]).to(x_1.device)
    
    def twoside_cfm_loss(self,network:nn.Module,x_0:torch.Tensor,x_1:torch.Tensor,*args,**kwargs) -> torch.Tensor:
        """
        Return the two-sided conditional flow matching loss

        Args:
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned velocity field
            x_0 (torch.Tensor): The samples from the source distribution. The first dimension should be the batch size
            x_1 (torch.Tensor): The samples from the target distribution. The first dimension should be the batch size
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
            
        Returns:
            torch.Tensor: The two-sided conditional flow matching loss
        """
        t=self.sample_t(x_1)
        x_t = self.psi_t(x_0,x_1,t)
        v_t = self.velocity(x_0,x_1,t)
        return torch.mean((network(x_t,t,*args,**kwargs)-v_t)**2)
    
    def cfm_loss(self,network:nn.Module,x_1:torch.Tensor,*args,**kwargs) -> torch.Tensor:
        """
        Return the conditional flow matching loss

        Args:
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned velocity field
            x_1 (torch.Tensor): The samples from the target distribution. The first dimension should be the batch size
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
        
        Returns:
            torch.Tensor: The conditional flow matching loss
        """
        return self.twoside_cfm_loss(network,torch.randn_like(x_1),x_1,*args,**kwargs)

    def sample(
        self,
        x_0:torch.tensor,
        network:nn.Module,
        solver_config:Union[FixedStepConfig,AdaptiveStepConfig],
        full_trajectory:bool=False,
        *args,
        **kwargs
    ) -> torch.tensor:
        """
        Generate samples from the flow

        Args:
            x_0 (torch.tensor): The initial samples. The first dimension should be the batch size
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned velocity field
            solver_config (Union[FixedStepConfig,AdaptiveStepConfig]): The configuration for the ODE solver
            full_trajectory (bool): If True, the full trajectory will be returned. Default to False
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural

        Returns:
            torch.tensor: The samples from the flow
        """
        def wrapper(t,x):
            return network(x,
                           t*torch.ones(x.shape[0],1).to(x_0.device),
                           *args,
                           **kwargs)
        return odeint(
            f=wrapper,
            x_0=x_0,
            t_0=0.,
            t_1=1.,
            solver_config=solver_config,
            record_trajectory=full_trajectory
        )

    def fixed_step_sample(
        self,
        x_0:torch.tensor,
        network:nn.Module,
        num_steps:int,
        solver:ODESolver=ODESolver.DOPRI45,
        full_trajectory :bool =False,
        *args,
        **kwargs) -> torch.tensor:
        """
        Generate samples from the flow using fixed step size
        
        Args:
            x_0 (torch.tensor): The initial samples. The first dimension should be the batch size
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned velocity field
            num_steps (int): The number of steps
            solver (ODESolver): The ODE solver. Default to ODESolver.DOPRI45
            full_trajectory (bool): If True, the full trajectory will be returned. Default to False
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
        
        Returns:
            torch.tensor: The samples from the flow
        
        """
        return self.sample(
            x_0=x_0,
            network=network,
            solver_config=FixedStepConfig(solver=solver,dt=1/num_steps),
            full_trajectory=full_trajectory,
            *args,
            **kwargs
        )
    
    def adaptive_sample(
        self,
        x_0:torch.tensor,
        network:nn.Module,
        solver:ODESolver=ODESolver.DOPRI45,
        atol:float=1e-6,
        rtol:float=1e-5,
        full_trajectory:bool=False,
        *args,
        **kwargs
    ) -> torch.tensor:
        """
        Generate samples from the flow using adaptive step size
        
        Args:
            x_0 (torch.tensor): The initial samples. The first dimension should be the batch size
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned velocity field
            solver (ODESolver): The ODE solver. Default to ODESolver.DOPRI45
            atol (float): The absolute tolerance for adaptive time stepping. Default to 1e-6
            rtol (float): The relative tolerance for adaptive time stepping. Default to 1e-5
            full_trajectory (bool): If True, the full trajectory will be returned. Default to False
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
        
        Returns:
            torch.tensor: The samples from the flow
        """
        return self.sample(
            x_0=x_0,
            network=network,
            solver_config=AdaptiveStepConfig(solver=solver,atol=atol,rtol=rtol),
            full_trajectory=full_trajectory,
            *args,
            **kwargs
        )

class OTFlowMatcher(ConditionalFlowMatcher):
    """
    The optimal transport flow matcher    
    """
    
    def __init__(self,sig_min:float=0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1-(1-self.sig_min)*t)*x_0 + t*x_1
    
    def velocity(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1-(1-self.sig_min)*x_0
        
class VEDifFlowMatcher(ConditionalFlowMatcher):
    
    def __init__(self) -> None:
        super().__init__()


    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.T(t))
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(1. - t) * x_1 + torch.sqrt(1. - self.alpha(1. - t) ** 2) * x_0
    
    def velocity(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt=self.psi_t(x_0,x_1,t)
        num = torch.exp(-self.T(1. - t)) * xt - torch.exp(-0.5 * self.T(1.-t))* x_1
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)