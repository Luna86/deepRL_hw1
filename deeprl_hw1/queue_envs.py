# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 3 * 5 * 5 * 5
        self.nA = 4
        self.p = np.array([p1, p2, p3])

    def _reset(self):
        self.state = (1,0,0,0)
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        return (1,0,0,0) 

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        reward = 0
        if action == QueueEnv.SERVICE_QUEUE and self.state[self.state[0]] > 0:
            reward = 1
            self.state[self.state[0]] -= 1
        for i in range(3):
            rand = np.random.rand()
            if rand < self.p[i]:
                self.state[i+1] = min(self.state[i+1] + 1, 5)
        return self.state, reward, False, None

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        reward = 0
        if action == QueueEnv.SERVICE_QUEUE:
            if state[state[0]] > 0:
                reward = 1
                state[state[0]] -= 1
        else:
            state[0] = action
        outcome = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    outcome.append(abs(i-self.p[0])*abs(j-self.p[1])*abs(k-self.p[2]), (state[0], min(state[1]+1-i, 5),
                    min(state[2]+1-j, 5),  min(state[3]+1-k, 5)), reward, False)
        return outcome

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
