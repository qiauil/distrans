from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .odesolve import odeint, ODESolver, FixedStepConfig, AdaptiveStepConfig
from typing import Union, Optional

class ConditionalFlowMatcher(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self._unifrom_sampler = torch.distributions.uniform.Uniform(0.,1.)
    
    @abstractmethod
    def psi_t(self,x_0:torch.Tensor,x_1:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        """ push forward function that maps samples $x_0$ from $p_0$ to $x_t$ """
    
    @abstractmethod
    def u_t(self,x_0:torch.Tensor,x_1:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        """ u_t field of the flow """
        
    def sample_t(self,x_1:torch.tensor) -> torch.tensor:
        """
        Sample time index from uniform distribution

        Args:
            x_1 (torch.tensor):Samples from the target distribution. 
                The shape of the returned tensor is [x_1.shape[0],1]

        Returns:
            torch.tensor: Samples of time index
        """
        return self._unifrom_sampler.sample([x_1.shape[0]]+[1]*(x_1.dim()-1)).to(x_1.device)
       
    def cfm_loss(self,network:nn.Module,
                 x_1:torch.Tensor,
                 x_0:Optional[torch.Tensor]=None,
                 *args,**kwargs) -> torch.Tensor:
        """
        Return the conditional flow matching loss

        Args:
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned u_t field
            x_1 (torch.Tensor): The samples from the target distribution. The first dimension should be the batch size
            x_0 (torch.Tensor): The samples from the source distribution. The first dimension should be the batch size. Default to None
                If None, the samples will be generated from the standard Gaussian distribution. Note that not all flow matchers support un-Gaussian $x_0$
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
        
        Returns:
            torch.Tensor: The conditional flow matching loss
        """
        x_0 = torch.randn_like(x_1) if x_0 is None else x_0
        t=self.sample_t(x_1)
        x_t = self.psi_t(x_0,x_1,t)
        v_t = self.u_t(x_0,x_1,t)
        return torch.mean((network(x_t,t.view(t.shape[0]),*args,**kwargs)-v_t)**2)        

    def sample(
        self,
        network:nn.Module,
        x_0:torch.tensor,
        solver_config:Union[FixedStepConfig,AdaptiveStepConfig],
        full_trajectory:bool=False,
        *args,
        **kwargs
    ) -> torch.tensor:
        """
        Generate samples from the flow

        Args:
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned u_t field
            x_0 (torch.tensor): The initial samples. The first dimension should be the batch size
            solver_config (Union[FixedStepConfig,AdaptiveStepConfig]): The configuration for the ODE solver
            full_trajectory (bool): If True, the full trajectory will be returned. Default to False
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural

        Returns:
            torch.tensor: The samples from the flow
        """
        def wrapper(t,x):
            return network(x,
                           t*torch.ones((x.shape[0],)).to(x_0.device),
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
        network:nn.Module,
        x_0:torch.tensor,
        num_steps:int,
        solver:ODESolver=ODESolver.DOPRI45,
        full_trajectory :bool =False,
        *args,
        **kwargs) -> torch.tensor:
        """
        Generate samples from the flow using fixed step size
        
        Args:
            x_0 (torch.tensor): The initial samples. The first dimension should be the batch size
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned u_t field
            num_steps (int): The number of steps
            solver (ODESolver): The ODE solver. Default to ODESolver.DOPRI45
            full_trajectory (bool): If True, the full trajectory will be returned. Default to False
            *args: Additional arguments for the neural network
            **kwargs: Additional keyword arguments for the neural network
        
        Returns:
            torch.tensor: The samples from the flow
        
        """
        return self.sample(
            network=network,
            x_0=x_0,
            solver_config=FixedStepConfig(solver=solver,dt=1/num_steps),
            full_trajectory=full_trajectory,
            *args,
            **kwargs
        )
    
    def adaptive_sample(
        self,
        network:nn.Module,
        x_0:torch.tensor,
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
            network (nn.Module): A neural network that takes $x_t$ and $t$ as input and returns the conditioned u_t field
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
            network=network,
            x_0=x_0,
            solver_config=AdaptiveStepConfig(solver=solver,atol=atol,rtol=rtol),
            full_trajectory=full_trajectory,
            *args,
            **kwargs
        )

class OTCondFlowMatcher(ConditionalFlowMatcher):
    """
    The optimal transport conditional flow matcher
    Only support $x_0$ to be standard Gaussian distribution
    
    Paper:
        [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
        
    Args:
        sig_min (float): The minimum value of the noise level. Default to 0.001
    """
    
    def __init__(self,sig_min:float=0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1-(1-self.sig_min)*t)*x_0 + t*x_1
    
    def u_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1-(1-self.sig_min)*x_0
        
class VPDifCondFlowMatcher(ConditionalFlowMatcher):
    """
    The Variance Preserving (VP) diffusion path conditional flow matcher
    The current implementation only supports the standard Gaussian distribution as the source distribution.
    The current implementation use a linear noise schedule. To use other noise schedules, please override the `beta` and `T` functions.

    Modified from `https://github.com/gle-bellier/flow-matching/`

    Paper:
        [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
        [Denoising diffusion probabilistic models.](https://arxiv.org/abs/2006.11239)
    
    Args:
        beta_min (float): The minimum value of the noise level. Default to 0.1
        beta_max (float): The maximum value of the noise level. Default
    """
    
    
    def __init__(self,beta_min:float=0.1,beta_max=20.0) -> None:
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Noise schedule
        """
        return self.beta_min + t*(self.beta_max - self.beta_min)
    
    def T(self, t: torch.Tensor) -> torch.Tensor:
        """
        Integration of the noise schedule
        """
        return self.beta_min * t + 0.5 * (t ** 2) * (self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.T(t))
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(1. - t) * x_1 + torch.sqrt(1. - self.alpha(1. - t) ** 2) * torch.randn_like(x_1)
    
    def u_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt=self.psi_t(x_0,x_1,t)
        num = torch.exp(-self.T(1. - t)) * xt - torch.exp(-0.5 * self.T(1.-t))* x_1
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)

class VEDiffusionCondFlowMatching(ConditionalFlowMatcher):

    """
    The Variance Exploding (VE) diffusion path conditional flow matcher
    The current implementation only supports the standard Gaussian distribution as the source distribution.
    The current implementation use a linear noise schedule. To use other noise schedules, please override the `sig_t` and `dsig_dt` functions.

    Modified from `https://github.com/gle-bellier/flow-matching/`

    Paper:
        [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
        [Generative modeling by estimating gradients of the data distribution](https://arxiv.org/abs/1907.05600)
    
    Args:
        sig_min (float): The minimum value of the noise level. Default to 0.01
        sig_max (float): The maximum value of the noise level. Default to 20.0
    """

    def __init__(self,sig_min=0.01, sig_max=20.0) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.sig_max = sig_max

    def sig_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Noise schedule
        """
        return self.sig_min * (self.sig_max / self.sig_min) ** t

    def dsig_dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the noise schedule
        """
        return self.sig_t(t) * torch.log(torch.tensor(self.sig_max/self.sig_min))
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1 + self.sig_t(1. - t) * torch.randn_like(x_1)
    
    def u_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -(self.dsig_dt(1. - t) / self.sig_t(1. - t)) * (x_0 - x_1)
    
class TwoSideOTCondFlowMatcher(ConditionalFlowMatcher):
    """
    Optimal transport conditional flow matcher with two side conditions, i.e., both $x_0$ and $x_1$ can from arbitrary distributions
    
    Paper:
        [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482)

    Args:
        sig_min (float): The minimum value of the noise level. Default to 0.001
    """

    def __init__(self,sig_min:float=0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t*x_1+(1-t)*x_0+self.sig_min*torch.randn_like(x_0)
    
    def u_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1-x_0
    
class TwoSideSchrodingerBridgeCondFlowMatcher(ConditionalFlowMatcher):

    """
    Schrodinger bridge conditional flow matcher with two side conditions, i.e., both $x_0$ and $x_1$ can from arbitrary distributions
    
    Paper:
        [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482)

    Args:
        sig_min (float): The minimum value of the noise level. Default to 0.001
    """

    def __init__(self,sig_min:float=0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
    
    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t*x_1+(1-t)*x_0+self.sig_min*torch.sqrt(t*(1-t))*torch.randn_like(x_0)
    
    def u_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_t = self.psi_t(x_0,x_1,t)
        return (1-2*t)/(2*t*(1-t))*(x_t-(t*x_1+(1-t)*x_0))+(x_1-x_0)
    