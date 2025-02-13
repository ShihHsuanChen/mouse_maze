import abc
from copy import deepcopy
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np


def set_random_state(random_state: int):
    np.random.seed(random_state)


@dataclass
class State:
    board: np.ndarray
    position: Tuple[int,int]
    end: bool


Action = int

class MouseMaze:
    def __init__(
            self,
            dim: Tuple[int,int] = (2,3),
            cheese_size_num: Dict[int,int] = {1:1, 3:1}, # {size: number}
            cheese_size_reward: Dict[int,float] = {},
            num_poisons: int = 1,
            poison_reward: float = -100,
            goal_reward: float = 0,
            border_reward: float = 0,
            random_state: Optional[int] = None,
            ):
        self.dim = dim
        self.cheese_size_num = deepcopy(cheese_size_num)
        self.num_poisons = num_poisons
        self.cheese_size_reward = deepcopy(cheese_size_reward)
        self.poison_reward = poison_reward
        self.goal_reward = goal_reward
        self.border_reward = border_reward
        # object map
        # s: start, g: goal, p: poison, c-N: cheese of size N
        self.obj_cnt_map = {'s': 1, 'g': 1, 'p': num_poisons}
        self.obj_cnt_map.update({
            f'c-{size}': num 
            for size, num in cheese_size_num.items()
        })
        self.obj_reward_map = {'s': 0, 'g': goal_reward, 'p': poison_reward}
        self.obj_reward_map.update({
            f'c-{size}': cheese_size_reward.get(size, size)
            for size in cheese_size_num.keys()
        })
        self.idx_obj_map: Dict[int,str] = dict(enumerate(self.obj_cnt_map.keys(), start=1))
        self.obj_idx_map: Dict[str,int] = {v:k for k,v in self.idx_obj_map.items()}
        self.idx_cnts_map: Dict[int,int] = {
            k: self.obj_cnt_map[obj]
            for k, obj in self.idx_obj_map.items()
        }
        # validate
        obj_cnts = sum(self.obj_cnt_map.values())
        if obj_cnts > dim[0] * dim[1]:
            raise ValueError(
                f"Board dimension too small! Requires at least \
                #cheeses + #poisons + 2 ({obj_cnts})"
            )

        if random_state is not None:
            set_random_state(random_state)
        self.random_state = random_state

        self._board, self._st_pt = self._create_game_board(dim, self.idx_cnts_map)
        self.reset()

    def _create_game_board(
            self,
            dim: Tuple[int,int],
            idx_cnts_map: Dict[int,int]
            ) -> Tuple[np.ndarray, Tuple[int,int]]:
        j = 0
        board = np.zeros(dim[0]*dim[1], dtype=np.uint32)
        for idx, cnts in idx_cnts_map.items():
            board[j:j+cnts] = idx
            j += cnts
        np.random.shuffle(board)
        board = board.reshape(dim)
        st_pts = np.argwhere(board==1)
        if len(st_pts) == 0:
            raise ValueError('Cannot find start point')
        elif len(st_pts) > 1:
            raise ValueError('Multiple start points')
        st = st_pts[0]
        return board, (int(st[0]), int(st[1]))

    def reset(self):
        self._state = State(
            board=np.copy(self._board),
            position=self._st_pt,
            end=False,
        )

    @property
    def state(self):
        return self._state

    def _move(self, position: Tuple[int,int], action: Action) -> Tuple[int,int]:
        # actions: up: 0, right: 1, down: 2, left: 3
        i,j = self._state.position
        if action == 0:
            i -= 1
        elif action == 1:
            j += 1
        elif action == 2:
            i += 1
        elif action == 3:
            j -= 1
        return (i,j)

    def step(self, action: Action) -> float:
        board = self._state.board
        i, j = self._move(self._state.position, action)
        if not (0 <= i < board.shape[0]):
            return self.border_reward
        if not (0 <= j < board.shape[1]):
            return self.border_reward
        self._state.position = (i,j)
        obj = self.idx_obj_map.get(board[i,j])
        print(obj)
        if obj is None:
            return 0
        elif obj in ['g', 'p']:
            self._state.end = True
        else:
            self._state.board[i,j] = 0
        return self.obj_reward_map.get(obj, 0)

    def display(self, state: Optional[State] = None):
        if state is None:
            state = self._state
        board = self._state.board
        dim = board.shape
        w = 5
        blank_line = '|'.join([' '*w for _ in range(dim[1])])
        border_line = '|'.join(['_'*w for _ in range(dim[1])])
        for i in range(dim[0]):
            if i == 0:
                print((' '+'_'*w)*dim[1])
            line = '|'.join([
                self.idx_obj_map.get(board[i,j], '').center(w)
                if (i,j) != state.position else 'm'.center(w)
                for j in range(dim[1])
            ])
            print(f'|{blank_line}|')
            print(f'|{line}|')
            print(f'|{border_line}|')

    @property
    def game_over(self):
        return self._state.end


class MouseMazeAgent(abc.ABC):
    @abc.abstractmethod
    def move(self, state: State) -> Action:
        raise NotImplementedError()
