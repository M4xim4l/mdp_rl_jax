from jax import random
import jax.numpy as jnp

class MDP:
    def __init__(self, key, num_states, num_actions, num_rewards = 5, reward_mean=0.0, reward_std=1.0):
        self.num_states = num_states
        self.num_actions = num_actions

        self.reward_mean = reward_mean
        self.reward_std = reward_std

        #finite set of possible rewards
        key, subkey = random.split(key)
        self.rewards = reward_std * random.normal(subkey, (num_rewards,), dtype=jnp.float32) + reward_mean
        del subkey

        #transitions probabilities:
        #transition_ps[s,a,s'] = probability of ending in state s' when executing action a at state s
        key, subkey = random.split(key)
        transition_ps = random.uniform(subkey, (num_states, num_actions, num_states), dtype=jnp.float32)
        del subkey

        self.transition_ps = transition_ps / transition_ps.sum(axis=-1, keepdims=True)

        #reward probabilities:
        #reward_ps[s,a,r_i] = probability of getting reward self.rewards[r_i] when executing action a at state s
        key, subkey = random.split(key)
        reward_ps = random.uniform(subkey, (num_states, num_actions, num_rewards), dtype=jnp.float32)
        del subkey

        self.reward_ps = reward_ps / reward_ps.sum(axis=-1, keepdims=True)

        #expected rewards [s,a]
        self.expected_rewards = jnp.sum(self.reward_ps * self.rewards[None, None, :], axis=2)











