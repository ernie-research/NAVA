"""
@longbin ji, personalized scheduler for diffusion training
"""
import torch, math

class SchedulerConfig:
    """
    scheduler configuration class
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        """
        get method
        """
        return self.__dict__.get(key, default)

    def __getattr__(self, name):
        return self.__dict__.get(name)


class FlowMatchScheduler():
    """
    FlowMatchScheduler class
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)
        self.config = SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )


    def set_timesteps(
            self, 
            num_inference_steps=100, 
            denoising_strength=1.0, 
            training=False, 
            shift=None, 
            dynamic_shift_len=None, 
            exponential_shift_mu=None
        ):
        """
        set_timesteps function
        """
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """
        step function for sampling
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """
        return to timestep
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        """
        add noise for single element
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def add_noise_batch(self, original_samples, noise, timesteps):
        """
        original_samples: [B, C, ...]
        noise:            [B, C, ...]
        timesteps:        [B]
        """

        device = original_samples[0].device if isinstance(original_samples, list) else original_samples.device
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=device) # support tensor and list input

        timesteps = timesteps.to(self.timesteps.device)

        # [T] -> [1, T]
        # [B] -> [B, 1]
        dist = (self.timesteps[None, :] - timesteps[:, None]).abs()
        timestep_ids = dist.argmin(dim=1)        # [B]

        sigmas = self.sigmas[timestep_ids].to(device)       # [B]

        if isinstance(original_samples, list):
            sample = []
            for i, (x, n) in enumerate(zip(original_samples, noise)):
                s = sigmas[i]  # scalar tensor
                while s.ndim < x.ndim:
                    s = s.unsqueeze(-1)
                sample.append((1 - s) * x + s * n)
            return sample
        else:
            # reshape for broadcast
            while sigmas.ndim < original_samples.ndim:
                sigmas = sigmas.unsqueeze(-1)

            sample = (1 - sigmas) * original_samples + sigmas * noise
            return sample
    

    def training_target(self, sample, noise, timestep):
        """
        get training target
        """
        if isinstance(sample, list):
            target = [n - s for s, n in zip(sample, noise)]
        else:
            target = noise - sample
        return target
    

    # def training_weight(self, timestep):
    #     timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
    #     weights = self.linear_timesteps_weights[timestep_id]
    #     return weights
    def training_weight(self, timestep):
        """
        training reweight function
        """
        timestep = timestep.to(self.timesteps.device)

        # [B, T]
        dist = (self.timesteps[None, :] - timestep[:, None]).abs()
        timestep_id = dist.argmin(dim=1)      # [B]

        weights = self.linear_timesteps_weights[timestep_id]  # [B]
        return weights
    
    
    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len=256,
        max_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
    ):
        """
        calculate shift
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
