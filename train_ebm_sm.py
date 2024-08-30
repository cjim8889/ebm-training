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
import functools


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


def denoising_score_matching_loss(model, x, key, noise_level):
    noise = jax.random.normal(key, x.shape) * noise_level
    x_noisy = x + noise
    scores = jax.vmap(stein_score, in_axes=(None, 0))(model, x_noisy)
    loss = 0.5 * jnp.mean(
        jnp.linalg.norm(
            jnp.reshape(scores + (noise / noise_level**2), (scores.shape[0], -1)),
            axis=1,
            ord=2,
        )
        ** 2
    )
    return loss


def sliced_score_matching_loss(model, x, key, num_slices=5):
    batch_size, channels, height, width = x.shape
    x = x.reshape(batch_size, -1)

    vectors = jax.random.normal(
        key, (num_slices, batch_size, channels * height * width)
    )
    vectors = vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)

    def score_and_jacobian(x):
        score = stein_score(model, x.reshape(-1, channels, height, width))
        return score.reshape(batch_size, -1)

    score_fn = jax.vmap(score_and_jacobian)

    def compute_loss(v):
        score, jvp = jax.jvp(score_fn, (x,), (v,))
        loss_1 = jnp.sum(jvp * v, axis=-1)
        loss_2 = 0.5 * jnp.sum(score**2, axis=-1)
        return loss_1 + loss_2

    losses = jax.vmap(compute_loss)(vectors)
    return jnp.mean(losses)


def sample_langevin_dynamics(key, model, x, num_steps=60, step_size=10):
    def step(i, x):
        noise = jax.random.normal(jax.random.fold_in(key, i), shape=x.shape) * 0.005
        x = x + noise
        x = jnp.clip(x, -1.0, 1.0)

        grad = stein_score(model, x)
        grad = jnp.clip(grad, -0.03, 0.03)

        x = x - step_size * grad
        x = jnp.clip(x, -1.0, 1.0)
        return x

    return jax.lax.fori_loop(0, num_steps, step, x)


@eqx.filter_jit
def sample_images(key, model, num_samples=16):
    z = jax.random.normal(key, (num_samples, 3, 32, 32))
    samples = jax.vmap(sample_langevin_dynamics, in_axes=(0, None, 0))(
        jax.random.split(key, num_samples), model, z
    )
    return samples


def train(
    model, train_loader, optim, key, epochs, print_every, noise_level, loss_type="dsm"
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x, key):
        if loss_type == "dsm":
            loss_fn = functools.partial(
                denoising_score_matching_loss, noise_level=noise_level
            )
        elif loss_type == "ssm":
            loss_fn = sliced_score_matching_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, key)
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
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
            "noise_level": args.noise_level,
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
    optim = optax.adam(args.lr, b1=0.0)  # No momentum

    # Train the model
    train(
        model,
        train_loader,
        optim,
        key,
        args.epochs,
        args.print_every,
        args.noise_level,
        loss_type=args.loss_type,
    )


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
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Noise level for denoising score matching",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dsm",
        choices=["dsm", "ssm"],
        help="Loss type: denoising score matching (dsm) or sliced score matching (ssm)",
    )

    args = parser.parse_args()
    main(args)
