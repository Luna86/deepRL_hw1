#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.lake_envs as lake_env
import gym
import time
from rl import *

def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps

def run_optimal_policy(env, gamma, policy):
    state = env.reset()
    #env.render()
    #time.sleep(0.5)
    total_reward = 0
    num_steps = 0
    while True:
        state, reward, is_terminal, debug_info = env.step(policy[state])
        #env.render()
        total_reward += gamma**num_steps*reward
        num_steps += 1
        if is_terminal:
            break
        #time.sleep(0.5)
    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    #env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    #env = gym.make('Deterministic-4x4-FrozenLake-v0')
    grid_size=4

    print_env_info(env)
    print_model_info(env, 0, lake_env.DOWN)
    print_model_info(env, 1, lake_env.DOWN)
    print_model_info(env, 14, lake_env.RIGHT)
    
    gamma = 0.9
    tol = 1e-3
    input('Hit enter to run value iteration...')
    
    start_time = time.time()
    value_func, iter_idx = value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3)
    time_elapsed = time.time() - start_time
    
    print('Time for execution of value iteration: {0}'.format(time_elapsed))
    print('Number of value iteration steps: {0}'.format(iter_idx))
    #print('Value function by value iteration: {0}'.format(value_func))
    print('Value function by value iteration: ')
    #print(value_func)
    for i in range(grid_size):
        #print(" ".join([str(round(x,4)) for x in value_func[i*grid_size:(i+1)*grid_size]]) + ";")
        print(" & ".join([str(round(x,4)) for x in value_func[i*grid_size:(i+1)*grid_size]]) + "\\\\")
        #print(" ".join([str(x) for x in value_func[i*grid_size:(i+1)*grid_size]]) + ";")
    policy = value_function_to_policy(env, gamma, value_func)
    print_policy(policy, lake_env.action_names)

    input('Hit enter to run the optimal agent')
    total_iter = 100
    total_reward = 0
    for i in range(total_iter):
        reward, num_steps = run_optimal_policy(env, gamma, policy)
        total_reward += reward
    print('Agent received total reward of: %f' % total_reward)
    #print('Agent took %d steps' % num_steps)

if __name__ == '__main__':
    main()
