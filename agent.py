import numpy as np

from game import MouseMazeAgent, State, Action


class Agent(MouseMazeAgent):
    def move(self, state: State) -> Action:
        return np.random.randint(4)
