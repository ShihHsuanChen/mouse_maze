import os
import time
from game_gym import MouseMazeEnv
from agent import Agent

gamma = 0.9

def main():
    game = MouseMazeEnv()
    agent = Agent()
    value = 0
    obs, _ = game.reset()
    game.render()
    while True:
        action = agent.move(obs)
        obs, reward, terminated, _, _ = game.step(action)
        value = gamma * value + reward
        os.system('clear')
        game.render()
        print('Reward', reward, 'Value:', value)
        if terminated:
            break
        time.sleep(1)


if __name__ == '__main__':
    main()
