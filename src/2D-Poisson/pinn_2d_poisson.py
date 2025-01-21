import jax, flax, optax, time, pickle
import os
import jax.numpy as jnp
from functools import partial
import json
from pyDOE import lhs
from typing import Sequence
from tensorflow_probability.substrates import jax as tfp
import numpy as onp

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

ARCHITECTURE_LIST = [[60,60,60,1]]
lr = 1e-3
num_epochs = 20000

class PDESolution(flax.linen.Module):
    features: Sequence[int]

    @flax.linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = flax.linen.tanh(flax.linen.Dense(feat)(x))
        x = flax.linen.Dense(self.features[-1])(x)
        return x

@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
@jax.jit
def analytic_sol(xs,ys):
    out = (xs**2) * ((xs-1)**2) * ys * ((ys-1)**2)
    return out


@jax.jit
def analytic_sol1(xs,ys):
    out = (xs**2) * ((xs-1)**2) * ys * ((ys-1)**2)
    return out

@partial(jax.vmap, in_axes=(None, 0, 0,), out_axes=(0, 0, 0))
@jax.jit
def neumann_derivatives(params,xs,ys):
    u = lambda x, y: model.apply(params, jnp.stack((x, y)))
    du_dx_0 = jax.jvp(u,(0.,ys),(1.,0.))[1]
    du_dx_1 = jax.jvp(u,(1.,ys),(1.,0.))[1]
    du_dy_1 = jax.jvp(u,(xs,1.),(0.,1.))[1]
    return du_dx_0, du_dx_1, du_dy_1

# PDE residual for 2D Poisson
@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x, y):
    H1 = jax.hessian(u, argnums=0)(x,y)
    H2 = jax.hessian(u, argnums=1)(x,y)
    lhs = H1+H2
    rhs = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) + (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y )
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x, y: model.apply(params, jnp.stack((x, y))), points[:, 0], points[:, 1])**2)


@partial(jax.jit, static_argnums=0)
def pde_true(analytic_sol,params, points):
    return jnp.mean((model.apply(params, jnp.stack((points[:, 0], points[:, 1]), axis=1)) - analytic_sol(points[:, 0], points[:, 1]) )**2)

@jax.jit
def boundary_dirichlet(params, points): 
    return jnp.mean((model.apply(params, jnp.stack((points[:,0],jnp.zeros_like(points[:,1])), axis=1)))**2)

@partial(jax.jit, static_argnums=0) 
def boundary_neumann(neumann_derivatives, params, points):
    du_dx_0, du_dx_1, du_dy_1 = neumann_derivatives(params,points[:,0],points[:,1])
    return jnp.mean((du_dx_0)**2) + jnp.mean((du_dx_1)**2) + jnp.mean((du_dy_1)**2)

@partial(jax.jit, static_argnums=(1, 4))
def training_step(params, opt, opt_state, key, neumann_derivatives):
    lb = jnp.array([0.,0.])
    ub = jnp.array([1.,1.])
    domain_points = lb + (ub-lb)*lhs(2, 2000)
    boundary = lb + (ub-lb)*lhs(2, 250)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) +
                                                    boundary_dirichlet(params, boundary) +
                                                    boundary_neumann(neumann_derivatives, params, boundary))(params)

    loss_pde = pde_residual(params, domain_points)
    loss_bc_d = boundary_dirichlet(params, boundary)
    loss_bc_n = boundary_neumann(neumann_derivatives, params, boundary)

    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key, loss_val


def train_loop(params, adam, opt_state, key, neumann_derivatives):
    losses = []
    for _ in range(num_epochs):
        params, opt_state, key, loss_val= training_step(params, adam, opt_state, key, neumann_derivatives)
        losses.append(loss_val.item())
    return losses, params, opt_state, key, loss_val

def concat_params(params):
        params, tree = jax.tree_util.tree_flatten(params)
        shapes = [param.shape for param in params]
        return jnp.concatenate([param.reshape(-1) for param in params]), tree, shapes

def unconcat_params(params, tree, shapes):
        split_vec = jnp.split(params, onp.cumsum([onp.prod(shape) for shape in shapes]))
        split_vec = [vec.reshape(*shape) for vec, shape in zip(split_vec, shapes)]
        return jax.tree_util.tree_unflatten(tree, split_vec)

for n, feature in enumerate(ARCHITECTURE_LIST):
    for _ in range(1):
        model = PDESolution(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        batch_dim = 8
        feature_dim = 2
        params = model.init(key, jnp.ones((batch_dim, feature_dim)))
        adam = optax.adam(lr)
        opt_state = adam.init(params)
        losses, params, opt_state, key, loss_val = jax.block_until_ready(train_loop(params, adam, opt_state, key, neumann_derivatives))

        lb = jnp.array([0.,0.])
        ub = jnp.array([1.,1.])
        domain_points = lb + (ub-lb)*lhs(2, 1000)
        boundary = lb + (ub-lb)*lhs(2, 25)

        init_point, tree, shapes = concat_params(params)

        results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_points) +
                                                            boundary_dirichlet(unconcat_params(params, tree, shapes), boundary) +
                                                            boundary_neumann(neumann_derivatives,unconcat_params(params, tree, shapes), boundary)),
                                    init_point,
                                    max_iterations=50000,
                                    num_correction_pairs=50,
                                    f_relative_tolerance=1.0 * jnp.finfo(float).eps)

        tuned_params = unconcat_params(results.position, tree, shapes)

        X_, Y_ = onp.meshgrid(onp.linspace(0, 1, 1000), onp.linspace(0, 1, 1000))
        domain_points = onp.vstack([X_.flatten(), Y_.flatten()]).T

        u_approx = model.apply(tuned_params, jnp.stack((domain_points[:, 0], domain_points[:, 1]), axis=1)).squeeze()

        u_true = analytic_sol(domain_points[:,0],domain_points[:,1]).squeeze()
        run_accuracy = onp.linalg.norm(u_approx - u_true)/onp.linalg.norm(u_true)

    u_true = u_true.tolist()
    u_pinn, domain_pts, l2_rel = u_approx.tolist(), domain_points.tolist(), run_accuracy.tolist()
    n+=1
    results = {
        'evaluation_points': domain_pts,
        'u_pinn': u_pinn,
        'u_true': u_true,
        'l2_rel': l2_rel
    }
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "pinn_2d_poisson.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
    print(l2_rel)
    print(f"Results saved to '{output_path}'.")