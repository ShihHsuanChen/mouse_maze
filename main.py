import os
import time
from game import MouseMaze
from agent import Agent

gamma = 0.9

def main():
    game = MouseMaze(random_state=None)
    agent = Agent()
    value = 0
    game.display()
    while not game.game_over:
        action = agent.move(game.state)
        reward = game.step(action)
        value = gamma * value + reward
        os.system('clear')
        game.display()
        print('Reward', reward, 'Value:', value)
        time.sleep(1)


if __name__ == '__main__':
    main()
