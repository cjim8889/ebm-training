from einops import rearrange, repeat

from .dem_module import *
from .components.mcmc import time_batched_sample_hamiltonian_monte_carlo

class LiouvilleModule(DEMLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
        time_schedule: BaseTimeSchedule = None,
    ) -> None:
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            energy_function=energy_function,
            noise_schedule=noise_schedule,
            lambda_weighter=lambda_weighter,
            buffer=buffer,
            num_init_samples=num_init_samples,
            num_estimator_mc_samples=num_estimator_mc_samples,
            num_samples_to_generate_per_epoch=num_samples_to_generate_per_epoch,
            num_samples_to_sample_from_buffer=num_samples_to_sample_from_buffer,
            num_samples_to_save=num_samples_to_save,
            eval_batch_size=eval_batch_size,
            num_integration_steps=num_integration_steps,
            lr_scheduler_update_frequency=lr_scheduler_update_frequency,
            nll_with_cfm=nll_with_cfm,
            nll_with_dem=nll_with_dem,
            nll_on_buffer=nll_on_buffer,
            logz_with_cfm=logz_with_cfm,
            cfm_sigma=cfm_sigma,
            cfm_prior_std=cfm_prior_std,
            use_otcfm=use_otcfm,
            nll_integration_method=nll_integration_method,
            use_richardsons=use_richardsons,
            compile=compile,
            prioritize_cfm_training_samples=prioritize_cfm_training_samples,
            input_scaling_factor=input_scaling_factor,
            output_scaling_factor=output_scaling_factor,
            clipper=clipper,
            score_scaler=score_scaler,
            partial_prior=partial_prior,
            clipper_gen=clipper_gen,
            diffusion_scale=diffusion_scale,
            cfm_loss_weight=cfm_loss_weight,
            use_ema=use_ema,
            use_exact_likelihood=use_exact_likelihood,
            debug_use_train_data=debug_use_train_data,
            init_from_prior=init_from_prior,
            compute_nll_on_train_data=compute_nll_on_train_data,
            use_buffer=use_buffer,
            tol=tol,
            version=version,
            negative_time=negative_time,
            num_negative_time_steps=num_negative_time_steps,
        )

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        loss = 0.0
        print(self.training_data.shape)
        exit()

    def generate_samples(self, initial_samples, ts):
        """
        Generate samples using the Euler method.
            t = 0 -> t = 1 : noise -> data
        """
        samples = initial_samples
        t_prev = ts[:-1]
        t_next = ts[1:]

        samples_list = [initial_samples]
        for t_p, t_n in zip(t_prev, t_next):
            t = torch.ones(samples.size(0), device=self.device).unsqueeze(1) * t_p
            with torch.no_grad():
                samples = samples + self.net(t, samples) * (t_n - t_p)
            samples_list.append(samples)
    
        samples = torch.stack(samples_list, dim=0)
        return samples


    def generate_samples_with_hmc_corrector(self):
        initial_samples = self.prior.sample(self.num_samples_to_generate_per_epoch)
        ts = self.time_schedule()

        samples = self.generate_samples(initial_samples, ts)
        final_samples = time_batched_sample_hamiltonian_monte_carlo(
            self.time_dependent_log_density,
            samples,
            ts,
            num_steps=5,
            integration_steps=5,
            eta=1.,
            rejection_sampling=False,
        )
        final_samples = rearrange(final_samples, 't n d -> n t d')
        return final_samples

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.prior = self.partial_prior(device=self.device)
        self.time_schedule = self.hparams.time_schedule
        self.net = self.net.to(self.device)

        # set up various functions
        self.time_dependent_log_density = lambda x, t: (1 - t) * self.prior.log_prob(x) + t * self.energy_function(x)

        if stage == "fit":
            # generate training data
            self.training_data = self.generate_samples_with_hmc_corrector()

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)
