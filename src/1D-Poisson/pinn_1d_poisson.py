import os
import json
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from pyDOE import lhs
from tensorflow_probability.substrates import jax as tfp
import numpy as onp
import flax.linen as nn

# Ensure GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using backend: {jax.default_backend()}")

# Hyperparameters
ARCHITECTURE_LIST = [[20, 20, 20, 1]]
LEARNING_RATE = 1e-6
NUM_EPOCHS = 15_000

class PDESolution(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        return nn.Dense(self.features[-1])(x)

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(lambda x: f(x)[0]), primals, tangents)[1]

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
def residual(u, x):
    v = jnp.ones_like(x)
    lhs = hvp(u, (x,), (v,))
    rhs = 16 * jnp.exp(-x**4) * x**7 - 20 * jnp.exp(-x**4) * x**3
    return lhs - rhs

@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x: model.apply(params, x), points) ** 2)

@jax.jit
def boundary_residual_0(params, xs):
    return jnp.mean(model.apply(params, jnp.zeros_like(xs)) ** 2)

@jax.jit
def boundary_residual_1(params, xs):
    return jnp.mean((model.apply(params, jnp.ones_like(xs)) - jnp.exp(-1.0)) ** 2)

@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):
    lb, ub = 0.0, 1.0
    domain_xs = lb + (ub - lb) * lhs(1, 256)
    boundary_xs = lb + (ub - lb) * lhs(1, 2)

    loss_val, grads = jax.value_and_grad(
        lambda p: pde_residual(p, domain_xs)
        + boundary_residual_0(p, boundary_xs)
        + boundary_residual_1(p, boundary_xs)
    )(params)

    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, loss_val

def train_loop(params, opt, opt_state, key):
    losses = []
    for _ in range(NUM_EPOCHS):
        params, opt_state, key, loss_val = training_step(params, opt, opt_state, key)
        losses.append(loss_val.item())
    return losses, params, opt_state

def concat_params(params):
    flat_params, tree = jax.tree_util.tree_flatten(params)
    shapes = [p.shape for p in flat_params]
    return onp.concatenate([p.ravel() for p in flat_params]), tree, shapes

def unconcat_params(params, tree, shapes):
    split_vec = onp.split(params, onp.cumsum([onp.prod(s) for s in shapes])[:-1])
    split_vec = [vec.reshape(shape) for vec, shape in zip(split_vec, shapes)]
    return jax.tree_util.tree_unflatten(tree, split_vec)

u_pinn, evaluation_points, l2_rel = {}, {}, {}

for n, feature in enumerate(ARCHITECTURE_LIST):
    print(f"Training model with architecture: {feature}")

    model = PDESolution(feature)
    key = jax.random.PRNGKey(17)

    params = model.init(key, jnp.ones((8, 1)))
    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(params)

    losses, params, opt_state = train_loop(params, opt, opt_state, key)

    lb, ub = 0.0, 1.0
    domain_xs = lb + (ub - lb) * lhs(1, 256)
    boundary_xs = lb + (ub - lb) * lhs(1, 2)

    init_point, tree, shapes = concat_params(params)

    results = tfp.optimizer.lbfgs_minimize(
        jax.value_and_grad(
            lambda p: pde_residual(unconcat_params(p, tree, shapes), domain_xs)
            + boundary_residual_0(unconcat_params(p, tree, shapes), boundary_xs)
            + boundary_residual_1(unconcat_params(p, tree, shapes), boundary_xs)
        ),
        init_point,
        max_iterations=50_000,
        num_correction_pairs=50,
        f_relative_tolerance=onp.finfo(float).eps,
    )

    tuned_params = unconcat_params(results.position, tree, shapes)
    domain_points = jnp.linspace(0, 1, 512).reshape(-1, 1)

    u_approx = model.apply(tuned_params, domain_points).squeeze()
    u_true = (domain_points * jnp.exp(-domain_points**4)).squeeze()

    l2_error = onp.linalg.norm(u_approx - u_true) / onp.linalg.norm(u_true)

    u_pinn[n] = u_approx.tolist()
    evaluation_points[n] = domain_points.tolist()
    l2_rel[n] = float(l2_error)

    print(f"L2 Relative Error: {l2_error:.2e}")

results = {
    "evaluation_points": evaluation_points,
    "u_pinn": u_pinn,
    "u_true": u_true.tolist(),
    "l2_rel": l2_rel,
}

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "pinn_1d_poisson.json")

with open(output_path, "w") as f:
    json.dump(results, f)

print(f"Results saved to '{output_path}'.")