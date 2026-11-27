import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import walker_sim as w_s
import IPython

goal_ball = 0.001
class WalkerEnv(gym.Env):
  def __init__(self):
    '''
    max_act = m-d array, where m is number of actuators
    max_obs is n-d array, where n is the state space of the robot.  Assumes 0 is the minimum observation
    init_state is the initial state of the entire robot
    '''
    max_act = np.ones(4) * 1000.0
    max_obs = np.ones(42) * 2.0
    goal_input = np.expand_dims(w_s.goal_pos, axis=0)
    
    
    
    self.action_space = spaces.Box(-max_act, max_act)
    self.observation_space = spaces.Box(np.zeros(42), max_obs)
    self.seed()
    
    self.iter_ = 0
    
    self.state = None
    self.goal_input = goal_input
    
    self.init_state, self.sim, self.loss, self.obs = w_s.generate_sim()
    self.sim.set_initial_state(initial_state=self.init_state)
    
  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
      
  def reset(self):
      self.state = self.init_state
      self.sim.set_initial_state(initial_state = self.init_state)  
      zero_act = np.expand_dims(np.zeros(self.action_space.shape), axis=0)
      #We need to step forward and get the observation
      #IPython.embed()
      memo_obs = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={w_s.goal: self.goal_input, w_s.actuation: zero_act},
        loss=self.obs)
          
      print('reset')
      return memo_obs.loss
      
  def step(self, action):        
    action_ = np.expand_dims(action, axis=0)
    #1. sim forward
    memo = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={w_s.goal: self.goal_input, w_s.actuation: action_},
        loss=self.loss)
        
    memo_obs = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={w_s.goal: self.goal_input, w_s.actuation: action_},
        loss=self.obs)
    
    if self.iter_ % 10 == 0:
      self.sim.visualize(memo, show=True)

        
    #2. update state
    #TODO: update state
    self.state = memo.steps[-1]
    
    obs = memo_obs.loss.flatten()
    
    
    #3. calculate reward as velocity toward the goal
    reward = memo.loss
    #print(reward)
    
    #TODO: 4. return if we're exactly at the goal and give a bonus to reward if we are
    success = np.linalg.norm(obs[18:20] - self.goal_input) < goal_ball #TODO: unhardcode
    if self.iter_ == 799:
      self.iter_ = -1
      fail = True
    else:
      fail = False
    
    if success:
      reward += 1
    elif fail:
      reward -= 1
      
    self.iter_ += 1
      
    done = fail or success
    
    self.memo = memo
            
    #print(reward)
    return obs, reward, done, {}
      
  def render(self):
    sim.visualize(self.memo, 1,show=True, export=w_s.exp, interval=1)
