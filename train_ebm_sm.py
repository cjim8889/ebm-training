import argparse

import numpy as np
import wandb
import torch
import torch.utils.data as data_utils
from torchvision.utils import make_grid
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from torchvision import transforms
from torchvision.datasets import CIFAR10


# Define CNN model using Equinox
class CNN(eqx.Module):
    layers: list
    input_channels: int
    hidden_features: int
    depth: int
    out_dim: int
    activation_fn: callable
    pool_type: str
    pool_every: int
    kernel_size: int
    stride: int
    padding: int
    final_pooling: bool
    input_size: tuple

    def __init__(
        self,
        key,
        input_channels=3,
        hidden_features=64,
        depth=4,
        out_dim=1,
        activation_fn=jax.nn.swish,
        pool_type="max",
        pool_every=2,
        kernel_size=3,
        stride=1,
        padding=1,
        final_pooling=True,
        input_size=(32, 32),
    ):
        self.input_channels = input_channels
        self.hidden_features = hidden_features
        self.depth = depth
        self.out_dim = out_dim
        self.activation_fn = activation_fn
        self.pool_type = pool_type
        self.pool_every = pool_every
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.final_pooling = final_pooling
        self.input_size = input_size

        keys = jax.random.split(key, depth + 2)
        self.layers = []
        in_channels = input_channels
        current_height, current_width = input_size

        for i in range(depth):
            out_channels = hidden_features * (2**i)
            self.layers.append(
                eqx.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    key=keys[i],
                )
            )
            self.layers.append(activation_fn)
            in_channels = out_channels

            current_height = (current_height - kernel_size + 2 * padding) // stride + 1
            current_width = (current_width - kernel_size + 2 * padding) // stride + 1

            if (i + 1) % pool_every == 0:
                if pool_type == "max":
                    self.layers.append(eqx.nn.MaxPool2d(kernel_size=2, stride=2))
                elif pool_type == "avg":
                    self.layers.append(eqx.nn.AvgPool2d(kernel_size=2, stride=2))
                current_height //= 2
                current_width //= 2

        if final_pooling:
            if pool_type == "max":
                self.layers.append(eqx.nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool_type == "avg":
                self.layers.append(eqx.nn.AvgPool2d(kernel_size=2, stride=2))
            current_height //= 2
            current_width //= 2

        self.layers.append(jnp.ravel)
        flattened_size = in_channels * current_height * current_width
        self.layers.append(eqx.nn.Linear(flattened_size, in_channels, key=keys[-2]))
        self.layers.append(activation_fn)
        self.layers.append(eqx.nn.Linear(in_channels, out_dim, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(axis=-1)


def stein_score(model, x):
    score_fn = jax.grad(model, argnums=0)
    return -score_fn(x)


def denoising_score_matching_loss(model, x, key):
    noise = jax.random.normal(key, x.shape) * 0.1
    x_noisy = x + noise
    scores = jax.vmap(stein_score, in_axes=(None, 0))(model, x_noisy)
    loss = 0.5 * jnp.mean(
        jnp.linalg.norm(
            jnp.reshape(scores + (noise / 0.1), (scores.shape[0], -1)), axis=1, ord=2
        )
        ** 2
    )
    return loss


def sample_langevin_dynamics(key, model, x, num_steps=10, step_size=0.01):
    def step(i, x):
        noise = jax.random.normal(jax.random.fold_in(key, i), shape=x.shape)
        grad = stein_score(model, x)
        x = x - (step_size / 2.0) * grad + jnp.sqrt(step_size) * noise
        return x

    return jax.lax.fori_loop(0, num_steps, step, x)


@eqx.filter_jit
def sample_images(key, model, num_samples=16):
    z = jax.random.normal(key, (num_samples, 3, 32, 32))
    samples = jax.vmap(sample_langevin_dynamics, in_axes=(0, None, 0))(
        jax.random.split(key, num_samples), model, z
    )
    return samples


def train(model, train_loader, optim, key, epochs, print_every):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x, key):
        loss_value, grads = eqx.filter_value_and_grad(denoising_score_matching_loss)(
            model, x, key
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    step = 0
    for epoch in range(epochs):
        for i, (x, _) in enumerate(train_loader):
            step += 1
            x = x.numpy()
            subkey, key = jax.random.split(key)
            model, opt_state, train_loss = make_step(model, opt_state, x, subkey)

            # Log metrics to WandB
            wandb.log({"train_loss": train_loss.item()}, step=step)

            if i % print_every == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {train_loss.item()}")

        subkey, key = jax.random.split(key)
        generated_samples = sample_images(subkey, model, num_samples=16)
        generated_samples = np.array(
            generated_samples
        )  # Ensure it's a writable NumPy array
        generated_samples = torch.tensor(generated_samples).cpu()

        grid = make_grid(generated_samples, nrow=4, normalize=True)
        wandb.log({"generated_samples": [wandb.Image(grid)]}, step=step)

    return model


def main(args):
    # Initialize WandB
    wandb.init(
        project="jax-cifar10",
        config={
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "print_every": args.print_every,
            "seed": args.seed,
            "cnn_hidden_features": args.hidden_features,
            "cnn_depth": args.depth,
            "optimizer": "adam",
        },
    )

    # Data transformation and loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = CIFAR10(
        root=args.dataset_path, train=True, transform=transform, download=True
    )
    train_loader = data_utils.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Initialize the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    model = CNN(
        key,
        hidden_features=args.hidden_features,
        depth=args.depth,
    )
    optim = optax.adam(args.lr)

    # Train the model
    train(model, train_loader, optim, key, args.epochs, args.print_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a JAX-based model on CIFAR-10")
    parser.add_argument(
        "--dataset_path", type=str, default="./data", help="Path to dataset"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--print_every", type=int, default=100, help="Print progress every N steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--hidden_features",
        type=int,
        default=64,
        help="Number of hidden features in CNN",
    )
    parser.add_argument("--depth", type=int, default=4, help="Depth of the CNN")

    args = parser.parse_args()
    main(args)
