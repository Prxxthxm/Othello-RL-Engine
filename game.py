from othello import Othello
import numpy as np


class Environment:
    def __init__(self):
        self.game = Othello()

    def get_state(self) -> tuple[np.ndarray, int]:
        """
        Retrieves the current state of the environment

        Returns:
            np.ndarray: the current board where 1 is a black piece, -1 is white, and 0 is an empty square.
            int: the ID of the player whose turn it is (+1/-1)
        """
        return self.game.get_input()

    def step(self, action: int) -> tuple[tuple[np.ndarray, int], int, int]:
        """
        Takes a step in the environment.

        Parameters
        ----------
        action : tuple of int
            The (row, column) coordinates where the agent places a piece.

        Returns
        -------
        next_state : tuple of (np.ndarray, int)
            The next state after taking the action. Includes the updated board and the current turn.
        reward : int
            The reward obtained from the action.
        done : bool
            True if the game is over after this action, otherwise False.
        """

        r, c = divmod(action, 8)
        player = self.game.current_turn

        success = self.game.take_turn(r, c)
        if not success:
            return self.get_state(), -1, 0

        done = int(self.game.is_game_over())
        reward = self._reward(player) if done else 0

        return self.get_state(), reward, done

    def _reward(self, player: int) -> int:
        winner = self.game.get_winner_id()
        if winner == 0:
            return 0

        return 1 if winner == player else -1

    def reset(self) -> None:
        self.game = Othello()