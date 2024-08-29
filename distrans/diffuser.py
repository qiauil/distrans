#usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union,Optional,Callable,Sequence


class Diffuser():
    """
    Diffuser class for diffusion models
    
    Args:
        betas (torch.Tensor): The diffusion schedule parameters.
        dim_data (int): The dimension of the data. Default to 2.
        device (Union[str,torch.device]): The device to store the tensors. Default to "cuda" if available, otherwise "cpu".
        dtype (torch.dtype): The data type of the tensors. Default to torch.float32.
    
    """
    def __init__(self, 
                 betas:torch.Tensor, 
                 dim_data:int=2,
                 device:Union[str,torch.device]="cuda" if torch.cuda.is_available() else "cpu",
                 dtype:torch.dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.steps = len(betas)      
        
        self.betas = torch.cat((torch.tensor([0]), betas), dim=0)  # set the first beta to 0
        self.betas = self.betas.view([self.steps+1, 1]+[1]*dim_data).to(self.device,dtype)
        self.alphas = 1-self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)

    def forward_diffusion(self, x_0:torch.Tensor, t:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion step
        
        Args:
            x_0 (torch.Tensor): The initial data.
            t (torch.Tensor): The time step.
            noise (torch.Tensor): The noise.
        
        Returns:
            torch.Tensor: The diffused $x_t$.
        """
        return self.sqrt_alphas_bar[t]*x_0+self.sqrt_one_minus_alphas_bar[t]*noise
    
    def DDPM_sample(self, 
                    model: Union[nn.Module, Callable], 
                    x_t: torch.Tensor,
                    show_progress=True, 
                    record_trajectory=False,
                    *args,
                    **kwargs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Sample from the DDPM model
        
        Args:
            model (Union[nn.Module, Callable]) The neural network.
            x_t (torch.Tensor): The initial data.
            show_progress (bool): Whether to show the progress bar. Default to True.
            record_trajectory (bool): Whether to record the trajectory. Default to False.
                If True, the function returns a list of states at each time step.
            *args: Additional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
            
        Returns: Union[torch.Tensor, Sequence[torch.Tensor]]: The samples from the DDPM model.
        """
        with torch.no_grad():
            if record_trajectory:
                samples = [x_t]
            t = torch.tensor([self.steps], device=self.device).repeat(x_t.shape[0])
            p_bar=range(self.steps)
            if show_progress:
                p_bar=tqdm(p_bar)
            for step in p_bar:
                noise = model(x_t, t, *args, **kwargs)
                x_t = self.DDPM_sample_step(x_t, t, noise)
                t = t-1
                if record_trajectory:
                    samples.append(x_t)
            if record_trajectory:
                return samples
            return x_t
        
    def DDIM_sample(self, 
                    model: Union[nn.Module, Callable], 
                    x_t: torch.Tensor,
                    dt: float,
                    show_progress=True, 
                    record_trajectory=False,
                    *args,
                    **kwargs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Sample from the DDIM model
        
        Args:
            model (Union[nn.Module, Callable]) The neural network.
            x_t (torch.Tensor): The initial data.
            dt (float): The time step.
            show_progress (bool): Whether to show the progress bar. Default to True.
            record_trajectory (bool): Whether to record the trajectory. Default to False.
                If True, the function returns a list of states at each time step.
            *args: Additional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
            
        Returns: Union[torch.Tensor, Sequence[torch.Tensor]]: The samples from the DDIM model.
        """
        with torch.no_grad():
            if record_trajectory:
                samples = [x_t]
            t = torch.tensor([self.steps], device=self.device).repeat(x_t.shape[0])
            p_bar=range(self.steps)
            if show_progress:
                p_bar=tqdm(p_bar)    
            while t[0] > 0:
                dt = min(dt, t[0])
                noise = model(x_t, t, *args, **kwargs)
                x_t = self.DDIM_sample_step(x_t, t,t-dt, noise)
                t = t-dt
                if record_trajectory:
                    samples.append(x_t)
            if record_trajectory:
                return samples
            return x_t    

    def DDPM_sample_step(self, x_t:torch.Tensor, t:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        """
        DDPM sample step
        
        Args:
            x_t (torch.Tensor): The data.
            t (torch.Tensor): The time index of `x_t`.
            noise (torch.Tensor): The noise.
        
        Returns:
            torch.Tensor: The denoised $x_{t-1}$.

        """
        coef1 = 1/self.sqrt_alphas[t]
        coef2 = self.betas[t]/self.sqrt_one_minus_alphas_bar[t]
        sig = torch.sqrt(self.betas[t])*self.sqrt_one_minus_alphas_bar[t-1]/self.sqrt_one_minus_alphas_bar[t]
        return coef1*(x_t-coef2*noise)+sig*torch.randn_like(x_t)

    def DDIM_sample_step(self, x_t, t, t_p, noise) -> torch.Tensor:
        """
        DDIM sample step
        
        Args:
            x_t (torch.Tensor): The data.
            t (torch.Tensor): The time index of `x_t`.
            t_p (torch.Tensor): The time index of the previous step.
            noise (torch.Tensor): The noise.
            
        Returns:
            torch.Tensor: The denoised $x_{t-dt}$.
        """
        with torch.no_grad():
            coef1 = 1/self.sqrt_alphas[t]
            coef2 = self.sqrt_one_minus_aplhas_bar[t] / \
                self.sqrt_alphas[t] - self.sqrt_one_minus_aplhas_bar[t_p]
            return coef1*x_t-coef2*noise

    def plot_paras(self):
        """
        Plot the diffusion parameters
        """
        plt.plot(self.betas[:,0,0,0].cpu(),label=r'$\beta$')
        plt.plot(self.alphas_bar[:,0,0,0].cpu(),label=r'$\bar{\alpha}$')
        plt.legend()
        plt.xlabel('$t$')
        plt.show()

    def to(self, device:Optional[Union[str,torch.device]]=None, dtype:Optional[torch.dtype]=None):
        """
        Move the diffuser to the specified device and data type
        
        Args:
            device (Optional[Union[str,torch.device]): The device to store the tensors.
            dtype (Optional[torch.dtype]): The data type of the tensors.
        """
        self.device = device if device is not None else self.device
        self.dtype = dtype if dtype is not None else self.dtype
        self.betas = self.betas.to(self.device,dtype)
        self.alphas = self.alphas.to(self.device,dtype)
        self.alphas_bar = self.alphas_bar.to(self.device,dtype)
        self.one_minus_alphas_bar = self.one_minus_alphas_bar.to(self.device,dtype)
        self.sqrt_alphas = self.sqrt_alphas.to(self.device,dtype)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(self.device,dtype)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(self.device,dtype)


class LinearParamsDiffuser(Diffuser):
    
    """
    Diffuser with linear diffusion schedule
    
    Args:
        steps (int): The number of steps.
        beta_min (float): The minimum beta.
        beta_max (float): The maximum beta.
        dim_data (int): The dimension of the data. Default to 2.
        device (Union[str,torch.device]): The device to store the tensors. Default to "cuda" if available, otherwise "cpu".
        dtype (torch.dtype): The data type of the tensors. Default to torch.float32.
    
    """

    def __init__(self, 
                 steps:int, 
                 beta_min:float, 
                 beta_max:float, 
                 dim_data:int=2,
                 device:Union[str,torch.device]="cuda" if torch.cuda.is_available() else "cpu",
                 dtype:torch.dtype=torch.float32
                 ):
        betas = torch.linspace(0, 1, steps)*(beta_max-beta_min)+beta_min
        super().__init__(betas, dim_data, device, dtype)


class sigParamsDiffuser(Diffuser):
    
    """
    Diffuser with sigmoid diffusion schedule
    
    Args:
        steps (int): The number of steps.
        beta_min (float): The minimum beta.
        beta_max (float): The maximum beta.
        dim_data (int): The dimension of the data. Default to 2.
        device (Union[str,torch.device]): The device to store the tensors. Default to "cuda" if available, otherwise "cpu".
        dtype (torch.dtype): The data type of the tensors. Default to torch.float32.
    """

    def __init__(self, 
                 steps:int, 
                 beta_min:float, 
                 beta_max:float, 
                 dim_data:int=2,
                 device:Union[str,torch.device]="cuda" if torch.cuda.is_available() else "cpu",
                 dtype:torch.dtype=torch.float32
                 ):
        betas = torch.sigmoid(torch.linspace(-6, 6, steps))*(beta_max-beta_min)+beta_min
        super().__init__(betas, dim_data, device, dtype)


class Cos2ParamsDiffuser(Diffuser):
    
    """
    diffuser with cosine diffusion schedule
    
    Args:
        steps (int): The number of steps.
        dim_data (int): The dimension of the data. Default to 2.
        device (Union[str,torch.device]): The device to store the tensors. Default to "cuda" if available, otherwise "cpu".
        dtype (torch.dtype): The data type of the tensors. Default to torch.float32.
    """

    def __init__(self,
                 steps:int, 
                 dim_data:int=2,
                 device:Union[str,torch.device]="cuda" if torch.cuda.is_available() else "cpu",
                 dtype:torch.dtype=torch.float32):
        s = 0.008
        tlist = torch.arange(1, steps+1, 1)
        temp1 = torch.cos((tlist/steps+s)/(1+s)*np.pi/2)
        temp1 = temp1*temp1
        temp2 = np.cos(((tlist-1)/steps+s)/(1+s)*np.pi/2)
        temp2 = temp2*temp2
        betas = 1-(temp1/temp2)
        betas[betas > 0.999] = 0.999
        super().__init__(betas, dim_data, device, dtype)