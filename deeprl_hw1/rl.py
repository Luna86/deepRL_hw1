# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
def action_value(env, gamma, value_function, s, a):
    q = 0
    for p in env.P[s][a]:
        #print('prob{0}: reward{1}: vs{2}'.format(p[0], p[2], value_function[p[1]]))
        q += p[0] * (p[2] + gamma * value_function[p[1]])
    return q

def check_terminal(env, s):
    is_terminal = True
    for a in range(len(env.P[s])):
        for p in env.P[s][a]:
            if p[1] != s:
                is_terminal = False
    return is_terminal

 

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    
    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_init = np.zeros(env.nS)
    eps = 1
    iter_idx = 0
    while eps > tol:
        eps = 0
        value = np.zeros(env.nS)
        #Bellman expecation equation backup
        for s in range(env.nS):
            #deterministic policy maps state to action
            action = policy[s]
            #transit stores all possible following (prob, state, reward, is_terminal) 
            transit = env.P[s][action]
            for p in transit:
                value[s] += p[0]*(p[2] + gamma * value_init[p[1]])
                eps = max(eps, abs(value[s] - value_init[s]))
        value_init = value
        iter_idx += 1
                
    return value_init, iter_idx 


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
        a_opt = 0
        qa_opt = action_value(env, gamma, value_function, s, a_opt)
        if check_terminal(env,s):
            continue
        for a in range(env.nA):
            qa = action_value(env, gamma, value_function, s, a)
            if qa > qa_opt:
                qa_opt = qa
                a_opt = a
        policy[s] = a_opt
    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_improve = value_function_to_policy(env, gamma, value_func)
    changed = not (np.array_equal(policy_improve, policy))
    return changed, policy_improve


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_iter_idx = 0
    value_iter_idx = 0
    converge = 0
    while policy_iter_idx < max_iterations and not converge:
        value_func, value_iter = evaluate_policy(env, gamma, policy, max_iterations, tol)
        changed, policy = improve_policy(env, gamma, value_func, policy)
        value_iter_idx += value_iter
        policy_iter_idx += 1
        converge = not changed

    return policy, value_func, policy_iter_idx, value_iter_idx


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    eps = 1
    iter_idx = 0
    value_init = np.zeros(env.nS)
    while eps > tol and iter_idx < max_iterations:
        eps = 0
        value = np.zeros(env.nS)
        for s in range(env.nS):
            if check_terminal(env, s):
                continue
            q_max = -np.inf
            a_idx = 0
            for a in range(env.nA):
                qa = action_value(env, gamma, value_init, s, a)
                #print('q value for S={0}, A={1}: {2}'.format(s, a, qa))
                if qa > q_max:
                    q_max = qa
                    a_idx = a
            eps = max(eps, abs(q_max - value_init[s]))
            value[s] = q_max
        value_init = value
        iter_idx += 1
        #print(value)
    return value_init, iter_idx


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
