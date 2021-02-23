#!/usr/bin/env python3

from gym.envs.registration import register

register(
  id='FingerEnv-v0',
  entry_point='finger.finger_env:FingerEnv',
)
