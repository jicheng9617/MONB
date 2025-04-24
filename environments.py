import numpy as np
from typing import Union


class MABEnv: 
    def __init__(self, 
                 num_arm: int = None, 
                 ) -> None:
        self.K = num_arm

    @property
    def num_arm(self): 
        return self.K 
    
    def get_reward(self, 
                   arm: int,
                   ) -> float: 
        """
        Get the reward for the arm

        Parameters
        ----------
        arm : int
            index of the selected arm

        Returns
        -------
        float
            reward value
        """
        raise NotImplementedError("Subclasses should implement this method.") 
    

class contextMABEnv(MABEnv): 
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 arm_context: np.ndarray = None, 
                 ) -> None:
        super().__init__(num_arm) 
        self.d = num_dim 
        self.A = np.array(arm_context) if isinstance(arm_context, list) else arm_context

    @property
    def arm_context(self) -> np.ndarray: 
        return self.A
    
    @property
    def num_dim(self) -> int: 
        return self.d
    
    def observe_context(self) -> np.ndarray: 
        """
        Output the arms' context at round $t$
        """
        raise NotImplementedError("Subclasses should implement this method.")

    
class contextMABSimulator(contextMABEnv): 
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 arm_context: np.ndarray = None,
                 ) -> None:
        super().__init__(num_arm, num_dim, arm_context)
    
    @property 
    def optimal_arm(self): 
        return self.opt_arm
    
    def reset(self): 
        raise NotImplementedError("Subclasses should implement this method.") 
        
    def observe_context(self) -> np.ndarray:
        if self.vary_context: 
            self._sample_context()
            self._eval_optimal()
        return self.arm_context
    
    def get_regret(self, arm: Union[int | np.ndarray | list]) -> np.ndarray: 
        """
        Get regrets for arms

        Parameters
        ----------
        arm : any
            arms' index, can be int, list or ndarray

        Returns
        -------
        np.ndarray
            regrets
        """
        if isinstance(arm, list): arm = np.array(arm)
        if isinstance(arm, np.ndarray): 
            return np.array(
                [self._eval_regret_arm(a_i) for a_i in arm]
            )
        else: 
            return self._eval_regret_arm(arm) 
            
    def _sample_context(self):
        """
        Generate arms' context randomly within unit sphere.

        Parameters
        ----------
        num_arm : int
            number of arms
        """
        self.A = np.zeros(shape=(self.K, self.d))
        for i in range(self.K): 
            unitVec = np.random.normal(size=self.d)
            unitVec /= np.linalg.norm(unitVec)
            self.A[i] = np.random.uniform() ** (1 / self.d) * unitVec
    
    def _eval_optimal(self): 
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _eval_regret_arm(self, arm):
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _eval_expected_reward(self, arm): 
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _h(self,
           arm: list | int | np.ndarray): 
        if isinstance(arm, int) or isinstance(arm, np.integer): 
            return self._eval_expected_reward(arm=self.A[arm])
        elif isinstance(arm, list): 
            return np.vstack([self._eval_expected_reward(arm=self.A[i]) for i in arm]).squeeze()
        elif isinstance(arm, np.ndarray): 
            arm = np.atleast_2d(arm) 
            return np.vstack([self._eval_expected_reward(arm=a) for a in arm])
    
    def _print_info(self): 
        """
        Print the information of the environment. 
        """
        print('# dimension: {}'.format(self.d))
        print('# arms: {}'.format(self.num_arm))
        print("Optimal arm: {}".format(self.optimal_arm))
        print('Regret for each arm: {}'.format(
            [round(self._eval_regret_arm(i),4) for i in range(self.num_arm)]
            ))
    

class moContextMABSimulator(contextMABSimulator):
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 num_obj: int = None, 
                 arm_context: np.ndarray = None,
                 obj_preference: str = 'scalarization', 
                 sclarization_type = 'weighted_sum',
                 opt_type: str = 'max', 
                 vary_context: bool = False, 
                 noise_var: float = 0.1, 
                 ) -> None:
        super().__init__(num_arm, num_dim, arm_context)
        self.m = num_obj 
        self.obj_preference = obj_preference
        self.sclarization_type = sclarization_type
        self.opt_type = opt_type
        self.vary_context = vary_context 
        self.R = noise_var 

    @property 
    def num_obj(self): 
        return self.m 
    
    def reset(self, 
              num_arm: int = None, 
              num_dim: int = None, 
              num_obj: int = None, 
              noise_var : float = None, 
              seed: int = None, 
              verbose: bool = False,
              ) -> None: 
        """
        Initialize the environment and sample the unknown parameters randomly. 
        """
        if num_arm is not None: self.K = num_arm 
        if num_dim is not None: self.d = num_dim 
        if num_obj is not None: self.m = num_obj 
        if noise_var is not None: self.R = noise_var 
        # check the setting
        assert self.K is not None, "Please assign number of arms!"
        assert self.d is not None, "Please define dimension of arms' context!" 
        assert self.m is not None, "Please set number of objectives!" 
        if (self.obj_preference.lower() in 'mpl-pc'+'mpl-pl') and self.priority is None: 
            raise NotImplementedError('Please assign the preference relationship between objectives!')
        self._sample_context()
        
    def observe_context(self) -> np.ndarray:
        if self.vary_context: 
            self._sample_context()
        return self.arm_context
    
    def _eval_optimal(self, weight_vector=None):
        self.expected_rewards = self._h(self.A)
        # evaluate the optimal indexs
        match self.obj_preference.lower():
            case 'scalarization': 
                self.opt_arm = np.argmax(self._scalarization(self.expected_rewards, weight_vector)) if self.opt_type.lower() == 'max' else np.argmin(self._scalarization(self.expected_rewards, weight_vector))

    def  _eval_regret_arm(self, arm, weight_vector):
        # get the expected reward for the arm
        arm_y = self._h(arm)
        # different regret type w.r.t. the preference setting
        match self.obj_preference.lower():
            case 'scalarization': 
                return self._scalarization(arm_y, weight_vector) - self._scalarization(self.expected_rewards[self.opt_arm], weight_vector) if self.opt_type.lower() == 'min' \
                    else self._scalarization(self.expected_rewards[self.opt_arm], weight_vector) - self._scalarization(arm_y, weight_vector)
    
    def _noise(self, size: int): 
        return np.random.normal(loc=0.0, scale=self.R, size=(size, self.m))
    
    def get_reward(self, arm: int) -> np.ndarray:
        if isinstance(arm, np.ndarray):
            return self._h(arm) + self._noise(size=len(arm))
        else: 
            return self._h(arm) + self._noise(size=1).squeeze()
        
    def _scalarization(self, 
                       y: np.ndarray, 
                       weight_vector: np.ndarray) -> np.ndarray: 
        y = np.atleast_2d(y)
        match self.sclarization_type.lower(): 
            case 'weighted_sum': 
                return np.sum(y*weight_vector, axis=1).reshape(-1, )
    
    def get_regret(self, 
                   arm: int | np.ndarray | list, 
                   weight_vector: np.ndarray = None) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        arm : int | np.ndarray | list
            _description_
        weight_vector : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        self._eval_optimal(weight_vector=weight_vector)
        if self.obj_preference.lower() == 'scalarization' and weight_vector is None:
            raise NotImplementedError('Please assign the preference vector between objectives!')
        if isinstance(arm, list): arm = np.array(arm)
        if isinstance(arm, np.ndarray): 
            return np.array(
                [self._eval_regret_arm(a_i, weight_vector) for a_i in arm]
            )
        else: 
            return self._eval_regret_arm(arm, weight_vector) 