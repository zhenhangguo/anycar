

# class MPPIParams:
#     sigma: float
#     gamma_mean: float
#     gamma_sigma: float
#     discount: float
#     sample_sigma: float
#     lam: float
#     n_rollouts: int
#     h_knot: int
#     a_min: jnp.ndarray
#     a_max: jnp.ndarray
#     a_mag: jnp.ndarray
#     a_shift: jnp.ndarray
#     delay: int
#     len_history: int
#     debug: bool
#     fix_history: bool
#     num_obs: int
#     num_actions: int
#     num_intermediate: int
#     spline_order: int
#     smooth_alpha: float = 0.8
#     dynamics: str = 'dbm'
#     dual: bool = False


# MPPIParams(
#         spline_order=2,
#         sigma=0.05,
#         gamma_sigma=0.0,
#         gamma_mean=1.0,
#         discount=1.0,
#         sample_sigma=1.0,
#         lam=0.1,
#         n_rollouts=600,        
#         a_min=[-1, -1.], # first dim steer, 2nd throttle
#         a_max=[1., 1.],
#         a_mag=[1., 1.],
#         a_shift=[0., 0.],
#         delay=0,
#         len_history=251,
#         debug=False,
#         fix_history=False,
#         num_obs=6,
#         num_actions=2,
#         num_intermediate=7,
#         h_knot=8,
#         smooth_alpha=1.0,
#         dynamics="transformer-jax",
#         dual=True, 
#         # dynamics="dbm",
#         # dual=False, 
#     )

import jax
import numpy as np
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


h_knot =  8
num_intermediate = 7
num_actions = 2
spline_order = 2
n_rollouts = 5
H = (h_knot -1 ) * num_intermediate + 1 
a_mean = jnp.zeros((H, num_actions))
a_mean_waypoint = a_mean[::7]
sigmas = jnp.array([0.05] * 2)
a_cov_per_step = jnp.diag(sigmas ** 2)
a_cov = jnp.tile(a_cov_per_step[None, :, :], (H, 1, 1))
# print(" sigmas =" + str(sigmas))
# print(" a_cov_per_step =" + str(a_cov_per_step))
# print(" a_cov =" + str(a_cov))
# print(" a_cov.shape =" + str(a_cov.shape))
a_mean_init = a_mean[-1:]
a_cov_init = a_cov[-1:]
action_sampled = jnp.zeros((n_rollouts, H, num_actions))
        
step_us = jnp.arange(H)
step_nodes = jnp.arange(h_knot) * (num_intermediate)

def node2u(nodes):
    spline = InterpolatedUnivariateSpline(step_nodes, nodes, k=spline_order)
    us = spline(step_us)
    return us

def u2node(us):
    spline = InterpolatedUnivariateSpline(step_us, us, k=spline_order)
    nodes = spline(step_nodes)
    return nodes

node2u_vmap = jax.vmap(node2u, in_axes=(0,))
u2node_vmap = jax.vmap(u2node, in_axes=(0,))

key_use, self_key = jax.random.split(jax.random.PRNGKey(123), 2)
key_use = jax.random.split(key_use, n_rollouts)

def single_sample(key, traj_mean, traj_cov):
    keys = jax.random.split(key, 8)

    print("traj_mean.shape = " + str(traj_mean.shape))
    print("traj_cov.shape = " + str(traj_cov.shape))
    print("keys.shape = " + str(key.shape))

    return jax.vmap(
        lambda key, mean, cov: jax.random.multivariate_normal(key, mean, cov)
    )(keys, traj_mean, traj_cov)

## Spline interpolation
a_mean_waypoint = a_mean_waypoint.at[:, 0].set(u2node(a_mean[:, 0]))
a_mean_waypoint = a_mean_waypoint.at[:, 1].set(u2node(a_mean[:, 1]))

print("a_mean_waypoint.shape = " + str(a_mean_waypoint.shape))

a_cov_waypoint = a_cov[::num_intermediate]

print("a_cov_waypoint.shape = " + str(a_cov_waypoint.shape))

a_sampled_waypoint = jax.vmap(single_sample, in_axes=(0, None, None))( # (N, h_knot, action_dim)
            key_use, a_mean_waypoint, a_cov_waypoint,
        )

print("a_sampled_waypoint.shape = " + str(a_sampled_waypoint.shape))

a_sampled = action_sampled.copy()
a_sampled = a_sampled.at[:, :, 0].set(node2u_vmap(a_sampled_waypoint[:, :, 0]))
a_sampled = a_sampled.at[:, :, 1].set(node2u_vmap(a_sampled_waypoint[:, :, 1]))

print("a_sampled.shape = " + str(a_sampled.shape))

# a_sampled_waypoint = jax.vmap(single_sample, in_axes=(0, None, None))( # (N, h_knot, action_dim)
#     key_use, a_mean_waypoint, a_cov_waypoint,
# )