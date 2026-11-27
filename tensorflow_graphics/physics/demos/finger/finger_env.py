import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import finger_sim as f_s
import IPython

goal_ball = 0.001
class FingerEnv(gym.Env):
  def __init__(self):
    '''
    max_act = m-d array, where m is number of actuators
    max_obs is n-d array, where n is the state space of the robot.  Assumes 0 is the minimum observation
    init_state is the initial state of the entire robot
    '''
    max_act = np.ones(2) * 4.0
    max_obs = np.ones(18) * 2.0
    
    
    self.multi_target = f_s.multi_target
    
    self.action_space = spaces.Box(-max_act, max_act)
    self.observation_space = spaces.Box(np.zeros(18), max_obs)
    self.seed()
    
    self.iter_ = 0
    
    self.state = None
    if not self.multi_target:
      self.goal_pos = np.array([0.6, 0.6])
    else:
      self.goal_pos = f_s.goal_pos
    if not self.multi_target:
      self.goal_range = np.zeros((2, ), dtype = np.float32)
    else:
      self.goal_range = f_s.goal_range
    
    self.init_state, self.sim, self.loss, self.obs = f_s.generate_sim()
    self.sim.set_initial_state(initial_state=self.init_state)
    self.epoch_ = 0
    import time
    self.tt1 = time.time()
    if self.multi_target:
      print('Multi target!!!!!!!!!!!!!!!!')
      self.fout = open('multi_target_simple.log', 'w')
    else:
      print('Single target!!!!!!!!!!!!!!!!')
      self.fout = open('single_target_simple.log', 'w')
    
  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
      
  def reset(self):
      import time
      tt2 = time.time()
      print(tt2 - self.tt1)
      self.tt1 = tt2
      self.epoch_ += 1
      self.state = self.init_state
      self.sim.set_initial_state(initial_state = self.init_state)  
      zero_act = np.expand_dims(np.zeros(self.action_space.shape), axis=0)
      #We need to step forward and get the observation
      # IPython.embed()
      self.goal_input = ((np.random.random(2) - 0.5) * self.goal_range + self.goal_pos)[None, :]
      memo_obs = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={f_s.goal: self.goal_input, f_s.actuation: zero_act},
        loss=self.obs)
          
      print('reset')
      return memo_obs.loss
      
  def step(self, action):        
    action_ = np.expand_dims(action, axis=0)
    #1. sim forward
    pre_point = self.state[0][:, :, -f_s.group_num_particles:].mean(axis = 2)
    '''
    memo = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={f_s.goal: self.goal_input, f_s.actuation: action_},
        loss=self.loss)
    '''
        
    memo_obs = self.sim.run(
        initial_state=self.state,
        num_steps=1,
        iteration_feed_dict={f_s.goal: self.goal_input, f_s.actuation: action_},
        loss=self.obs)
    
    # if self.epoch_ % 100 <= 5 and self.iter_ % 5 == 1:
    #   self.sim.visualize(memo_obs, show=True)

        
    #2. update state
    #TODO: update state
    self.state = memo_obs.steps[-1]
    # IPython.embed()
    now_point = self.state[0][:, :, -f_s.group_num_particles:].mean(axis = 2)
    
    obs = memo_obs.loss.flatten()
    
    
    #3. calculate reward as velocity toward the goal
    # print(now_point, self.goal_input)
    pre_dist = (((pre_point - self.goal_input) ** 2).sum(axis = 1) ** 0.5).sum(axis = 0)
    now_dist = (((now_point - self.goal_input) ** 2).sum(axis = 1) ** 0.5).sum(axis = 0)
    reward = (pre_dist - now_dist) * 10.# memo.loss
    # print('{:.8f}'.format(reward))
    
    #TODO: 4. return if we're exactly at the goal and give a bonus to reward if we are
    # success = np.linalg.norm(obs[12:14] - self.goal_input) < goal_ball #TODO: unhardcode

    self.iter_ += 1
      
    if self.iter_ == 80:
      self.iter_ = 0
      done = True
      print(self.epoch_, 'L2 distance: ', now_dist)
      print(self.epoch_, now_dist, file = self.fout)
    else:
      done = False
    #  fail = True
    # else:
    #  fail = False
    
    # if success:
    #   reward += 1
    # elif fail:
    #   reward -= 1
      
    # done = fail or success
    
    self.memo = memo_obs
            
    #print(reward)
    return obs, reward, done, {}
      
  def render(self):
    sim.visualize(self.memo, 1,show=True, export=f_s.exp, interval=1)
