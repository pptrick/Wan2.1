# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Plain first-order Euler solver for rectified-flow sampling.

The whole method is `x <- x + (sigma_next - sigma_now) * v`: the model's
velocity is taken as constant across the step. No history, no correction, no
predictor/corrector -- which makes it the natural baseline for the multistep
solvers in fm_solvers.py and fm_solvers_unipc.py.

The sigma schedule is built exactly as FlowUniPCMultistepScheduler builds it,
so a run swapped onto this solver differs only in the update rule.
"""
import math
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = ['FlowEulerScheduler']


class FlowEulerScheduler(SchedulerMixin, ConfigMixin):
    """Euler sampler for `prediction_type="flow_prediction"`.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Length of the training schedule.
        shift (`float`, defaults to 1.0):
            Static timestep shift applied when `use_dynamic_shifting` is False.
        use_dynamic_shifting (`bool`, defaults to False):
            Resolve the shift per call from `mu` instead of the config value.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 shift: Optional[float] = 1.0,
                 use_dynamic_shifting: bool = False,
                 prediction_type: str = "flow_prediction"):
        # Same construction as FlowUniPCMultistepScheduler: alphas descend to
        # 1/num_train_timesteps, so sigma tops out just short of pure noise.
        # That matters -- sigma == 1 gives alpha == 0 and log(0) downstream.
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting and shift is not None:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)

    def set_timesteps(self,
                      num_inference_steps: Union[int, None] = None,
                      device: Union[str, torch.device] = None,
                      sigmas: Optional[List[float]] = None,
                      mu: Optional[Union[float, None]] = None,
                      shift: Optional[Union[float, None]] = None):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min,
                                 num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        timesteps = sigmas * self.config.num_train_timesteps
        # Terminal sigma is 0: the last step lands exactly on clean data.
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self,
             model_output: torch.Tensor,
             timestep: Union[int, torch.Tensor],
             sample: torch.Tensor,
             return_dict: bool = True,
             generator=None):
        """One Euler step.

        `model_output` is the velocity v; for rectified flow the exact
        relation is v = eps - x0, so a step of d_sigma moves x by d_sigma * v.
        Taking v as constant over the step is the entire approximation.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output.to(
            sample.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
