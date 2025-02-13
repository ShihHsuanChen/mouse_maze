import abc
from copy import deepcopy
from typing import Tuple, Dict, Optional, List, Union

import gymnasium as gym
import numpy as np


ACTION_SPACE = gym.spaces.Discrete(4)
ActionType = Union[int, np.int64]


class MouseMazeEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 1,
    }

    def __init__(
        self,
        dim: Tuple[int, int] = (2, 3),
        cheese_size_num: Dict[int, int] = {1: 1, 3: 1},  # {size: number}
        cheese_size_reward: Dict[int, float] = {},
        num_poisons: int = 1,
        poison_reward: float = -100,
        goal_reward: float = 0,
        border_reward: float = 0,
    ):
        self.dim = dim
        self.cheese_size_num = deepcopy(cheese_size_num)
        self.num_poisons = num_poisons
        self.cheese_size_reward = deepcopy(cheese_size_reward)
        self.poison_reward = poison_reward
        self.goal_reward = goal_reward
        self.border_reward = border_reward
        # object map
        # m: mouse, g: goal, p: poison, c-N: cheese of size N
        self.obj_cnt_map = {"m": 1, "g": 1, "p": num_poisons}
        self.obj_cnt_map.update(
            {f"c-{size}": num for size, num in cheese_size_num.items()}
        )
        self.obj_reward_map = {"m": 0, "g": goal_reward, "p": poison_reward}
        self.obj_reward_map.update(
            {
                f"c-{size}": cheese_size_reward.get(size, size)
                for size in cheese_size_num.keys()
            }
        )
        self.idx_obj_map: Dict[int, str] = dict(
            enumerate(self.obj_cnt_map.keys(), start=1)
        )
        self.obj_idx_map: Dict[str, int] = {v: k for k, v in self.idx_obj_map.items()}
        self.idx_cnts_map: Dict[int, int] = {
            k: self.obj_cnt_map[obj] for k, obj in self.idx_obj_map.items()
        }
        # validate
        obj_cnts = sum(self.obj_cnt_map.values())
        if obj_cnts > dim[0] * dim[1]:
            raise ValueError(
                f"Board dimension too small! Requires at least \
                #cheeses + #poisons + 2 ({obj_cnts})"
            )

        # gym setting
        self.observation_space = gym.spaces.Space(self.dim, dtype=np.int32)
        self.action_space = ACTION_SPACE
        self._action_to_direction = {
            0: (-1,  0),  # up
            1: ( 0,  1),  # right
            2: ( 1,  0),  # down
            3: ( 0, -1),  # left
        }
        # state
        self.board: np.ndarray

    def _create_game_board(
        self,
        dim: Tuple[int, int],
        idx_cnts_map: Dict[int, int],
    ) -> np.ndarray:
        j = 0
        board = np.zeros(dim[0] * dim[1], dtype=np.uint32)
        for idx, cnts in idx_cnts_map.items():
            board[j : j + cnts] = idx
            j += cnts
        np.random.shuffle(board)
        return board.reshape(dim)

    def _get_positions(self, board: np.ndarray, obj: str) -> List[Tuple[int,int]]:
        idx = self.obj_idx_map.get(obj)
        if idx is None:
            raise ValueError('No object named "{obj}"')
        pts = np.argwhere(board == idx)
        return [(int(pt[0]), int(pt[1])) for pt in pts]

    def _get_position(self, board: np.ndarray, obj: str) -> Tuple[int,int]:
        pts = self._get_positions(board, obj)
        if len(pts) == 0:
            raise ValueError(f'Cannot find object "{obj}"')
        elif len(pts) > 1:
            raise ValueError(f'Multiple object "{obj}" were found')
        return pts[0]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.board = self._create_game_board(self.dim, self.idx_cnts_map)
        return self.board, {}

    def step(self, action: ActionType) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        i, j = self._get_position(self.board, 'm')
        di, dj = self._action_to_direction.get(int(action), (0,0))
        ni, nj = i + di, j + dj # next (i,j)
        # don't move
        if not (0 <= ni < self.board.shape[0]):
            return self.board, self.border_reward, False, False, {}
        if not (0 <= nj < self.board.shape[1]):
            return self.board, self.border_reward, False, False, {}
        obj = self.idx_obj_map.get(self.board[ni,nj])
        if obj is None:
            return self.board, 0, False, False, {}
        # move
        self.board[i,j] = 0
        self.board[ni,nj] = self.obj_idx_map['m']
        # reward
        reward = self.obj_reward_map.get(obj, 0)
        if obj in ['g', 'p']:
            return self.board, reward, True, False, {}
        else:
            return self.board, reward, False, False, {}

    def display(self, board: Optional[np.ndarray] = None):
        if board is None:
            board = self.board
        dim = board.shape
        w = 5
        blank_line = '|'.join([' '*w for _ in range(dim[1])])
        border_line = '|'.join(['_'*w for _ in range(dim[1])])
        for i in range(dim[0]):
            if i == 0:
                print((' '+'_'*w)*dim[1])
            line = '|'.join([
                self.idx_obj_map.get(board[i,j], '').center(w)
                for j in range(dim[1])
            ])
            print(f'|{blank_line}|')
            print(f'|{line}|')
            print(f'|{border_line}|')

    def render(self):
        self.display()


class MouseMazeAgent(abc.ABC):
    def move(self, observation: np.ndarray) -> ActionType:
        return ACTION_SPACE.sample()
