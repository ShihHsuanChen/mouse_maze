import numpy as np

from game_gym import MouseMazeAgent, ACTION_SPACE, ActionType


class Agent(MouseMazeAgent):
    def move(self, observation: np.ndarray) -> ActionType:
        return ACTION_SPACE.sample()
