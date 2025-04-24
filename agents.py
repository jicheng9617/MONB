import numpy as np 
import torch

import copy
from typing import Union

from models import ffNet


class MABAgent: 
    def __init__(self, 
                 num_arm: int=None, 
                 ) -> None:
        """_summary_

        Parameters
        ----------
        num_arm : int, optional
            _description_, by default None
        """
        if num_arm is not None: self.K = num_arm 

    @property 
    def num_arm(self): 
        return self.K
    
    @property 
    def round(self): 
        return self.t 
    
    @property
    def total_counts(self): 
        return np.sum(self.counts)
    
    @property
    def reward_history(self): 
        return np.array(self._reward_list)
    
    @property
    def action_history(self): 
        return np.array(self._action_list)
    
    def reset(self, 
              num_arm: int = None, 
              ) -> None:
        
        if num_arm is not None: self.K = num_arm
        assert self.K is not None, "Please assign the number of arms!"
        self.t = 0
        self.counts = np.zeros(self.K)
        self.mean_reward = np.zeros((self.K,))
        self._reward_list = []
        self._action_list = []

    def take_action(self, 
                    ) -> int: 
        return np.random.randint(0, self.K)

    def update(self, 
               info: Union[int, float]
               ) -> None: 
        """
        Update parameters in the algorithm.

        Parameters
        ----------
        info : Union[int, float]
            contains variables of action and reward
        """
        action, reward = info
        self.t += 1 
        self.mean_reward[action] = (self.mean_reward[action] * self.counts[action] + reward) / (self.counts[action] + 1)
        self.counts[action] += 1
        self._reward_list.append(reward)
        self._action_list.append(action)
        
        
class moMABAgent(MABAgent): 
    def __init__(self, 
                 num_arm = None, 
                 num_obj = None, 
                 ) -> None:
        super().__init__(num_arm)
        self.m = num_obj
        
    @property
    def num_obj(self): 
        return self.m
    
    def reset(self, 
              num_arm = None, 
              num_obj: int = None, 
              ) -> None:
        super().reset(num_arm)
        if num_obj is not None: self.m = num_obj
        self.mean_reward = np.zeros((self.K, self.m))

    def update(self, info):
        return super().update(info)


class moContextMABAgent(moMABAgent): 
    def __init__(self, 
                 num_dim: int, 
                 num_arm: int = None,
                 num_obj: int = None, 
                 lamda:float=1.,
                 delta:float=.05, 
                 ) -> None:
        super().__init__(num_arm, num_obj)
        self.d = num_dim 
        self.lamda = lamda 
        self.delta = delta 
        
    @property 
    def num_dim(self): 
        return self.d  
    
    @property 
    def context_his(self): 
        return np.vstack(self.X_list) 
    
    def reset(self, 
              num_arm: int = None, 
              num_obj: int = None, 
              num_dim: int = None, 
              ) -> None:
        super().reset(num_arm, num_obj)
        if num_dim is not None: self.d = num_dim
        if num_obj is not None: self.m = num_obj
        assert self.d is not None, "Please define dimension of arms' context!"
        self.X_list = [] 
        
    def take_action(self, 
                    context: np.ndarray) -> int:
        return super().take_action() 
    
    def update(self, 
               info: Union[int, float, np.ndarray]
               ) -> None:
        """
        Update parameters of the algorithm.

        Parameters
        ----------
        info : Union[int, float, np.ndarray]
            information containing the last chosen action, observed reward, and the context for that arm
        """
        action, reward, context = info
        super().update((action, reward))
        self.X_list.append(context)
        

class MONeural(moContextMABAgent):
    """
    Multi-objective neural bandits agents with UCB and TS
    """
    def __init__(self, 
                 num_dim: int = None, 
                 num_arm: int = None, 
                 num_obj: int = None, 
                 hidden_size: int = 100, 
                 hidden_layer: int = 1, 
                 rho: float = 1., 
                 lamda: float = 1, 
                 delta: float = 0.05, 
                 style: str = 'ucb', 
                 opt_type: str = 'max', 
                 obj_preference: str = 'scalarization', 
                 sclarization_type = 'weighted_sum',
                 lr: float = 5e-4, 
                 device: str = 'cpu', 
                 ) -> None:
        super().__init__(num_dim, num_arm, lamda, delta)
        self.m = num_obj
        self.rho = rho 
        self.style = style
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.opt_type = opt_type
        self.obj_type = obj_preference
        self.sclarization_type = sclarization_type
        self.lr = lr
        self.device = device

        self.gd_step = 2

    @property
    def num_obj(self): 
        return self.m
    
    def reset(self,
              num_dim: int = None,
              num_arm: int = None,
              num_obj: int = None,
              hidden: int = None,
              rho: float = None,
              lamda: float = None,
              delta: float = None,
              style: str = None,
              ) -> None:
        super().reset(num_arm=num_arm, num_dim=num_dim)
        if num_obj is not None: self.m = num_obj 
        if hidden is not None: self.hidden_size = hidden 
        if rho is not None: self.rho = rho 
        if lamda is not None: self.lamda = lamda 
        if delta is not None: self.delta = delta 
        if style is not None: self.style = style 
        assert self.m is not None, "#objectives not define!" 

        # initialize m neural networks for m objectives respectively
        self.f = [ffNet(dim=self.d, hidden_size=self.hidden_size, hidden_layer=self.hidden_layer).to(self.device) for _ in range(self.m)]
        self.total_param = [sum(p.numel() for p in func.parameters() if p.requires_grad) for func in self.f]
        self.U = [self.lamda * torch.ones((total_param,)).to(self.device) for total_param in self.total_param]
        self.update_threshold = 1.0
        self.U_prime = copy.deepcopy(self.U)
        self.optimizer = [torch.optim.Adam(net.parameters(), lr = self.lr, weight_decay=1e-3) for net in self.f] 

    def take_action(self, 
                    context: np.ndarray, 
                    weight_vector: list | np.ndarray, 
                    ) -> int:
        nonpull_arms = np.where(self.counts == 0)[0]
        if nonpull_arms.size > 0:  
            return np.random.choice(nonpull_arms)  
        else:
            tensor = torch.from_numpy(context).float().to(self.device)
            mu = torch.hstack([fx(tensor) for fx in self.f])
            samples = []
            for fx in mu: 
                g = self._eval_grad(fx)
                sigma2 = [self.lamda * g_i * g_i / U for g_i, U in zip(g, self.U)]
                sigma = torch.hstack([torch.sqrt(torch.sum(tmp)) for tmp in sigma2])
                match self.style.lower():
                    case 'ucb': sample_r = fx + self.rho * sigma
                    case 'ts': sample_r = torch.normal(mean=fx, std=self.rho * sigma)
                samples.append(sample_r)
            scalrized_samples = torch.sum(torch.vstack(samples) * torch.tensor(weight_vector).to(self.device), dim=1)
            arm = torch.argmax(scalrized_samples) if self.opt_type.lower() == 'max' else torch.argmin(scalrized_samples)
            return arm.item()
    
    def update(self, info: int | float | np.ndarray) -> None:
        super().update(info)
        action, reward, context = info
        # update U 
        c = torch.tensor(context).to(self.device)
        fx = torch.hstack([fx(c) for fx in self.f])
        g = self._eval_grad(torch.atleast_1d(fx))
        self.U = [self.U[i] + g[i]*g[i] for i in range(self.m)]
        # neural networks updates 
        if torch.max(torch.tensor([torch.sum(self.U[i]).item() / torch.sum(self.U_prime[i]).item() for i in range(self.m)])) >= self.update_threshold:
            # print("***************Network Updating***************")
            self.U_prime = copy.deepcopy(self.U)
            length = len(self.reward_history)
            index = np.arange(length) 
            np.random.shuffle(index)
            self._train(index)

    def _train(self, index): 
        for j in range(self.m): 
            self.f[j].train()
            for _ in range(self.gd_step): 
                self.optimizer[j].zero_grad() 
                y_pred = self.f[j](torch.tensor(np.vstack(self.X_list)).to(self.device)).reshape(-1, )
                y_true = torch.tensor(self.reward_history[:, j]).to(self.device)
                current_loss = torch.mean((y_pred - y_true) ** 2)
                if self.t == 0: 
                    current_loss.backward(retain_graph=True) 
                else: 
                    current_loss.backward()
                self.optimizer[j].step()

    def _scalarization(self, 
                       y: np.ndarray, 
                       weight_vector: np.ndarray) -> np.ndarray: 
        y = np.atleast_2d(y)
        match self.sclarization_type.lower(): 
            case 'weighted_sum': 
                return np.sum(y*weight_vector, axis=1).reshape(-1, )
            
    def _eval_grad(self, 
                   fx): 
        g = []
        for fx_i, func in zip(fx, self.f): 
            func.zero_grad()
            fx_i.backward(retain_graph=True)
            g.append(torch.cat([p.grad.flatten().detach() for p in func.parameters()]))
        return g
