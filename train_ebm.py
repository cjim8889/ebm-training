import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

# Path settings
DATASET_PATH = "./data"
CHECKPOINT_PATH = "./saved_models/ebm"

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set high precision for matrix multiplication to utilise Tensor Cores
torch.set_float32_matmul_precision("high")

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Configurable CNN Model
class ConfigurableCNNModel(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_features=64,
        depth=4,
        out_dim=1,
        activation_fn=Swish,
        pool_type="max",
        pool_every=2,
        kernel_size=3,
        stride=1,
        padding=1,
        final_pooling=True,
        input_size=(32, 32),
    ):
        super().__init__()

        cnn_layers = []
        in_channels = input_channels
        self.activation_fn = activation_fn()

        pooling_layer = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if pool_type == "max"
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

        current_height, current_width = input_size

        for i in range(depth):
            out_channels = hidden_features * (2**i)
            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            cnn_layers.append(self.activation_fn)
            in_channels = out_channels

            current_height = (current_height - kernel_size + 2 * padding) // stride + 1
            current_width = (current_width - kernel_size + 2 * padding) // stride + 1

            if (i + 1) % pool_every == 0:
                cnn_layers.append(pooling_layer)
                current_height //= 2
                current_width //= 2

        if final_pooling:
            cnn_layers.append(pooling_layer)
            current_height //= 2
            current_width //= 2

        cnn_layers.append(nn.Flatten())

        flattened_size = in_channels * current_height * current_width

        cnn_layers.append(nn.Linear(flattened_size, in_channels))
        cnn_layers.append(self.activation_fn)
        cnn_layers.append(nn.Linear(in_channels, out_dim))

        self.cnn_layers = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.cnn_layers(x).squeeze(dim=-1)


# Sampler class
class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [
            (torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)
        ]

    def sample_new_exmps(self, steps=60, step_size=10):
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(
            random.choices(self.examples, k=self.sample_size - n_new), dim=0
        )
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        inp_imgs = Sampler.generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size
        )

        self.examples = (
            list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.examples
        )
        self.examples = self.examples[: self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(
        model, inp_imgs, steps=60, step_size=10, return_img_per_step=False
    ):
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        noise = torch.randn_like(inp_imgs)
        imgs_per_step = []
        inp_imgs.requires_grad_(True)

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)

            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs


# Deep Energy Model
class DeepEnergyModel(pl.LightningModule):
    def __init__(
        self,
        model_class,
        img_shape,
        batch_size,
        alpha=0.1,
        lr=1e-4,
        beta1=0.0,
        **model_args,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = model_class(**model_args)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        return self.cnn(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999)
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        reg_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        self.log("loss", loss)
        self.log("loss_regularization", reg_loss)
        self.log("loss_contrastive_divergence", cdiv_loss)
        self.log("metrics_avg_real", real_out.mean())
        self.log("metrics_avg_fake", fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("val_contrastive_divergence", cdiv)
        self.log("val_fake_out", fake_out.mean())
        self.log("val_real_out", real_out.mean())


# Callbacks
class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size
        self.vis_steps = vis_steps
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            imgs_per_step = self.generate_imgs(pl_module)
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size - 1 :: step_size, i]
                grid = torchvision.utils.make_grid(
                    imgs_to_plot,
                    nrow=imgs_to_plot.shape[0],
                    normalize=True,
                    value_range=(-1, 1),
                )
                trainer.logger.log_image(
                    f"generation_{i}", [grid], step=trainer.current_epoch
                )

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(
            pl_module.device
        )
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)
        imgs_per_step = Sampler.generate_samples(
            pl_module.cnn,
            start_imgs,
            steps=self.num_steps,
            step_size=10,
            return_img_per_step=True,
        )
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step


class SamplerCallback(pl.Callback):
    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(
                random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0
            )
            grid = torchvision.utils.make_grid(
                exmp_imgs, nrow=4, normalize=True, value_range=(-1, 1)
            )
            trainer.logger.log_image("sampler", [grid], step=trainer.current_epoch)


class OutlierCallback(pl.Callback):
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand(
                (self.batch_size,) + pl_module.hparams["img_shape"]
            ).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar(
            "rand_out", rand_out, global_step=trainer.current_epoch
        )


def train_model(args):
    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=DATASET_PATH, train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATASET_PATH, train=False, transform=transform, download=True
    )

    train_loader = data_utils.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = data_utils.DataLoader(
        test_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # Initialize wandb
    wandb.init(project="ebm-cifar10", config=args)
    wandb_logger = WandbLogger(log_model="all")

    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "CIFAR10"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        gradient_clip_val=0.1,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_contrastive_divergence"
            ),
            GenerateCallback(every_n_epochs=1),
            SamplerCallback(every_n_epochs=1),
            # OutlierCallback(),
            LearningRateMonitor("epoch"),
        ],
    )

    # Create and train the model
    pl.seed_everything(args.seed)
    model = DeepEnergyModel(
        model_class=ConfigurableCNNModel,
        img_shape=(3, 32, 32),
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        hidden_features=args.hidden_features,
        depth=args.depth,
    )
    trainer.fit(model, train_loader, test_loader)

    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EBM on CIFAR-10")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--beta1", type=float, default=0.0, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        default=32,
        help="Number of hidden features in the CNN",
    )
    parser.add_argument("--depth", type=int, default=3, help="Depth of the CNN")
    parser.add_argument(
        "--max_epochs", type=int, default=120, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    model = train_model(args)
