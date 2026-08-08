#!/usr/bin/env python3

from gym.envs.registration import register

register(
  id='WalkerEnv-v0',
  entry_point='walker.walker_env:WalkerEnv',
)
