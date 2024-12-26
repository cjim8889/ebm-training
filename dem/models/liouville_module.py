from einops import rearrange, repeat

from torchmetrics import MeanMetric

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

        self.liouville_train_loss = MeanMetric()

    def get_loss(self, ts, xs):
        ts = ts.to(xs.device)
        ts = repeat(ts, 't -> n t', n=xs.size(0))

        dt_log_unormalised_density = self.time_derivative_log_density(xs, ts)
        dt_log_density = dt_log_unormalised_density - dt_log_unormalised_density.mean(dim=0, keepdim=True)

        score = self.score_function(xs, ts)

        b, t, d = xs.shape
        xs = rearrange(xs, "b t d -> (b t) d")
        ts = rearrange(ts, "b t -> (b t)")

        xs_detached = xs.detach().requires_grad_(True)
        v = self.net(ts.unsqueeze(-1), self.energy_function.normalize(xs_detached))

        div_v = torch.zeros(xs_detached.shape[:1], device=xs_detached.device)
        for i in range(xs_detached.shape[-1]):
            div_v += torch.autograd.grad(v[..., i].sum(), xs_detached, create_graph=True, retain_graph=True)[0][..., i]

        div_v = rearrange(div_v, "(b t) -> b t", b=b)
        v = rearrange(v, "(b t) d -> b t d", b=b)

        lhs = div_v + (v * score).sum(dim=-1)
        eps = (lhs + dt_log_density).nan_to_num_(posinf=1.0, neginf=-1.0, nan=0.0)
        return (eps**2).mean()

    def training_step(self, batch, batch_idx):
        loss = 0.0

        noised_samples = self.training_data[0][
            torch.randperm(self.training_data[0].size(0))[: self.num_samples_to_sample_from_buffer]
        ].detach()
        times = self.training_data[1]
        
        liouville_loss = self.get_loss(times, noised_samples)
        loss += liouville_loss

        self.liouville_train_loss(liouville_loss)
        self.log(
            "train/liouville_train_loss",
            self.liouville_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        self.training_data = self.generate_samples_with_hmc_corrector()

        initial_samples = self.prior.sample(self.num_samples_to_generate_per_epoch)
        ts = torch.linspace(0, 1, self.time_schedule.T)
        self.last_samples = self.generate_samples(initial_samples, ts)[-1, ...].detach()
        self.last_energies = self.energy_function(self.energy_function.normalize(self.last_samples))
        
        self._log_energy_w2(prefix="val")
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="val")
            self._log_dist_total_var(prefix="val")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        return

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self.eval_epoch_end("test")

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)

        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
        )

    def _log_energy_w2(self, prefix="val"):
        data_set = self.energy_function.sample_test_set(self.eval_batch_size)
        generated_energies = self.last_energies[: data_set.size(0)]

        energies = self.energy_function(self.energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_w2(self, prefix="val"):
        data_set = self.energy_function.sample_test_set(self.eval_batch_size)
        generated_samples = self.last_samples[: data_set.size(0)]

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_total_var(self, prefix="val"):
        data_set = self.energy_function.sample_test_set(self.eval_batch_size)
        generated_samples = self.last_samples[: data_set.size(0)]

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
        )
        data_set_dists = self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def score_function(self, x, t):
        x = x.detach().requires_grad_(True)
        log_p = self.time_dependent_log_density(x, t)
        return torch.autograd.grad(log_p.sum(), x)[0]

    def time_derivative_log_density(self, x, t=None):
        x = self.energy_function.normalize(x)
        return - self.prior.log_prob(x) + self.energy_function(x)

    def time_dependent_log_density(self, x, t):
        x = self.energy_function.normalize(x)
        return (1 - t) * self.prior.log_prob(x) + t * self.energy_function(x)

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
                samples = samples + self.net(t, self.energy_function.normalize(samples)) * (t_n - t_p)
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
        return (final_samples, ts)

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

        if stage == "fit":
            # generate training data
            self.training_data = self.generate_samples_with_hmc_corrector()

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)
