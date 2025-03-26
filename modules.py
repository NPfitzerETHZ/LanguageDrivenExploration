from typing import Any, Dict
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
import numpy as np
import math

def leaky_relu(x, key=None):
    return jax.nn.leaky_relu(x)

def default_init(key, linear, scale=1.0, zero_bias=False, fixed_bias=None):
    """Default init used in pytorch"""
    lim = math.sqrt(scale / linear.in_features)
    linear = eqx.tree_at(lambda l: l.weight, linear, jax.random.uniform(key, linear.weight.shape, minval=-lim, maxval=lim))
    if zero_bias:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.zeros_like(linear.bias))
    elif fixed_bias is not None:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.full_like(linear.bias, fixed_bias))
    return linear

def final_linear(key, input_size, output_size, scale=0.01):
    """a nn.Linear layer with initialization for the final layer of a value function"""
    #linear = ortho_linear(key, input_size, output_size, scale=scale)
    linear = nn.Linear(input_size, output_size, key=key)
    linear = default_init(key, linear, scale=scale, zero_bias=True)
    return linear

class Block(eqx.Module):
    """A standard nn layer with linear, norm, and activation."""
    net: eqx.Module
    def __init__(self, input_size, output_size, dropout, key):
        if dropout == 0.0:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                leaky_relu,
            ])
        else:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                nn.Dropout(dropout),
                leaky_relu,
            ])

    def __call__(self, x, key=None):
        return self.net(x, key=key)

class RandomSequential(nn.Sequential):
    """A nn.Sequential layer that passes through random keys"""
    def __call__(self, x, key=None):
        return super().__call__(x, key=key)

class QHead(eqx.Module):
    post0: eqx.Module
    post1: eqx.Module
    value: nn.Linear
    advantage: nn.Linear

    def __init__(self, input_size, hidden_size, output_size, dropout, key):
        keys = random.split(key, 3)

        self.post0 = Block(input_size, hidden_size, dropout, keys[0])
        self.post1 = Block(hidden_size, hidden_size, dropout, keys[1])
        self.value = final_linear(keys[2], hidden_size, 1, scale=0.01)
        self.advantage = final_linear(keys[3], hidden_size, output_size, scale=0.01)

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, 2)
        x = self.post0(x, net_keys[0])
        x = self.post1(x, net_keys[1])
        V = self.value(x) 
        A = self.advantage(x)
        # Dueling DQN
        return V + (A - A.mean(keepdims=True))


class GeneralQNetwork(eqx.Module):
    config: Dict[str, Any]
    q: eqx.Module
    reduce: callable

    def __init__(self, obs_size, task_size, act_size, config, key):
        self.config = config
        keys = jax.random.split(key)

        @eqx.filter_vmap
        def make_heads(key):
            return QHead(obs_size + task_size, config["head_size"], act_size, config["dropout"], key)
                    
        ensemble_keys = random.split(keys[0], config["ensemble_size"])
        self.q = make_heads(ensemble_keys)

        if config['ensemble_reduce'] == "median":
            self.reduce = jnp.median
        elif config['ensemble_reduce'] == "min":
            self.reduce = jnp.min
        else:
            raise Exception("Invalid reduce")
                    
    def __call__(self, x, task, key):
        """Returns an ensemble of Q values of shape [ensemble, actions]"""
        assert x.ndim == 1 and task.ndim == 1, "x dim: {}, task dim: {}".format(x.shape, task.shape)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
        def ensemble(model, x, key):
            return model(x, key=key)
        
        x = jnp.concatenate([x, task])
        q = ensemble(self.q, x, key)
        # Expects x to be of shape [S]
        return self.reduce(q, axis=0)


class SimpleGraphLayer(eqx.Module):
    W: nn.Linear

    def __init__(self, input_size, output_size, key):
        keys = random.split(key, 2)
        self.W = Block(2 * input_size, output_size, 0, key=keys[0])

    def conv(self, root, neighbors):
        # [A, F]
        # Discard index 0 because it is [root, root]
        res = eqx.filter_vmap(self.W)(jnp.concatenate([root, neighbors], axis=-1))
        # Agg
        return res.mean(0)


    def __call__(self, x):
        # x should be of shape [Num_agents, S]
        assert x.ndim == 2, "x dim: {}".format(x.shape)
        num_agents = x.shape[0]
        x = jnp.expand_dims(x, 1)
        roots = jnp.repeat(x, num_agents, axis=1)
        neighbors = roots.transpose(1, 0, 2)
        out = eqx.filter_vmap(self.conv)(roots, neighbors)
        return out


class GeneralMAQNetwork(eqx.Module):
    config: Dict[str, Any]
    gnn: SimpleGraphLayer
    q: eqx.Module
    debug: bool

    def __init__(self, obs_size, task_size, act_size, config, key, debug=False):
        self.config = config
        self.debug = debug
        keys = random.split(key, 3)

        if self.debug:
            self.gnn = None
            self.q = QHead(obs_size + task_size, config["head_size"], act_size, config["dropout"], keys[2])
        else:
            self.gnn = SimpleGraphLayer(obs_size + task_size, config["mlp_size"], keys[0])
            self.q = QHead(config["mlp_size"], config["head_size"], act_size, config["dropout"], keys[2])

                    
    def __call__(self, x, task, key):
        # x should be of shape [Num_agents, S]
        # TODO: Should we do relative pos/vel here?
        # We would need more memory (N^2) since neighbors would be different for each root
        assert x.ndim == 2 and task.ndim == 2, "x dim: {}, task dim: {}".format(x.shape, task.shape)
        net_keys = random.split(key, 3)
        x = jnp.concatenate([x, task], axis=-1)
        if not self.debug:
            x = self.gnn(x)
        q = eqx.filter_vmap(self.q)(x, random.split(net_keys[2], x.shape[0]))
        return q

def greedy_policy(
    q_network, x, task, key=None
):
    # Expand for ensemble
    q_values = q_network(x, task, key=key)
    action = jnp.argmax(q_values)
    return action

def epsilon_greedy_policy(
    q_network, x, task, epsilon, key=None,
):
    q_values = q_network(x, task, key=key)
    rand_action = random.randint(key, (1,), 0, q_values.shape[0])
    mask = random.uniform(key) < epsilon
    action = rand_action * mask + jnp.argmax(q_values) * ~mask
    return action.squeeze(0)
