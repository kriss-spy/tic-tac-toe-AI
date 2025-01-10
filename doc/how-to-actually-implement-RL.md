Your current implementation uses TensorFlow to build a neural network model, but it doesn't fully implement Reinforcement Learning (RL) as it lacks key components such as a proper reward system, policy updates, and a mechanism to learn from past experiences. Let's break down what you need to do to implement genuine RL in your Tic-Tac-Toe project.

### Key Components of RL

1. **Environment**: The Tic-Tac-Toe board and rules.
2. **Agent**: The player (X or O) that takes actions based on the current state.
3. **State**: The current configuration of the board.
4. **Action**: The move the agent makes (placing X or O on the board).
5. **Reward**: The immediate feedback from the environment after an action.
6. **Policy**: The strategy that the agent uses to decide actions based on the state.
7. **Value Function**: The expected long-term return of a state or action.

### Steps to Implement RL

1. **Define the Reward System**:

   - Assign rewards based on the outcome of the game.
   - For example:
     - Win: +1
     - Lose: -1
     - Tie: 0
2. **Update the Policy**:

   - Use a method like Q-learning or Policy Gradient to update the policy based on the rewards received.
3. **Experience Replay**:

   - Store past experiences (state, action, reward, next state) and use them to train the model.
4. **Exploration vs Exploitation**:

   - Use an epsilon-greedy strategy to balance exploration (random actions) and exploitation (actions based on the current policy).

### Implementing Q-Learning

Here’s a simplified version of how you can implement Q-learning in your Tic-Tac-Toe game:

1. **Initialize Q-table**:

   - Create a Q-table to store the expected rewards for each state-action pair.
2. **Choose Action**:

   - Use an epsilon-greedy strategy to choose actions.
3. **Update Q-values**:

   - Update the Q-values based on the reward received and the maximum future reward.
4. **Train the Model**:

   - Use the Q-values to train the neural network.

### Example Code

Here’s a modified version of your code to include Q-learning:

```python
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models

class ChessBoard:
    def __init__(self, size):
        self.size = size
        self.board_array = np.array([[None] * size for _ in range(size)])
        self.turn = 0

    def check(self):
        # Check rows, columns, and diagonals for a win
        for i in range(self.size):
            if self.board_array[i][0] and len(set(self.board_array[i])) == 1:
                return self.board_array[i][0]
            if self.board_array[0][i] and len(set([row[i] for row in self.board_array])) == 1:
                return self.board_array[0][i]
        if self.board_array[0][0] and len(set([self.board_array[i][i] for i in range(self.size)])) == 1:
            return self.board_array[0][0]
        if self.board_array[0][self.size - 1] and len(set([self.board_array[i][self.size - 1 - i] for i in range(self.size)])) == 1:
            return self.board_array[0][self.size - 1]
        return None

    def is_full(self):
        return all(all(cell is not None for cell in row) for row in self.board_array)

    def print_board(self):
        for row in self.board_array:
            print("|".join([cell if cell else " " for cell in row]))

class Agent:
    def __init__(self, char, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.char = char
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_state_key(self, state):
        return str(state)

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_key = self.get_state_key(state)
            q_values = [self.q_table.get((state_key, tuple(action)), 0) for action in valid_actions]
            max_q = max(q_values)
            if q_values.count(max_q) > 1:
                best_actions = [action for i, action in enumerate(valid_actions) if q_values[i] == max_q]
                return random.choice(best_actions)
            else:
                return valid_actions[q_values.index(max_q)]

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        next_max = max([self.q_table.get((next_state_key, tuple(a)), 0) for a in self.get_valid_actions(next_state)])
        self.q_table[(state_key, tuple(action))] = self.q_table.get((state_key, tuple(action)), 0) + \
            self.alpha * (reward + self.gamma * next_max - self.q_table.get((state_key, tuple(action)), 0))

    def get_valid_actions(self, state):
        return [(i, j) for i in range(len(state)) for j in range(len(state[i])) if state[i][j] is None]

class Game:
    def __init__(self):
        self.chess_board = ChessBoard(3)
        self.player1 = Agent('X')
        self.player2 = Agent('O')

    def run(self, episodes=1000):
        for episode in range(episodes):
            self.chess_board = ChessBoard(3)
            while True:
                state = self.chess_board.board_array
                valid_actions = self.player1.get_valid_actions(state)
                action = self.player1.choose_action(state, valid_actions)
                self.chess_board.board_array[action[0]][action[1]] = self.player1.char
                winner = self.chess_board.check()
                if winner:
                    reward = 1 if winner == self.player1.char else -1
                    self.player1.update_q_value(state, action, reward, self.chess_board.board_array)
                    self.player2.update_q_value(state, action, -reward, self.chess_board.board_array)
                    break
                if self.chess_board.is_full():
                    self.player1.update_q_value(state, action, 0, self.chess_board.board_array)
                    self.player2.update_q_value(state, action, 0, self.chess_board.board_array)
                    break

                state = self.chess_board.board_array
                valid_actions = self.player2.get_valid_actions(state)
                action = self.player2.choose_action(state, valid_actions)
                self.chess_board.board_array[action[0]][action[1]] = self.player2.char
                winner = self.chess_board.check()
                if winner:
                    reward = 1 if winner == self.player2.char else -1
                    self.player2.update_q_value(state, action, reward, self.chess_board.board_array)
                    self.player1.update_q_value(state, action, -reward, self.chess_board.board_array)
                    break
                if self.chess_board.is_full():
                    self.player2.update_q_value(state, action, 0, self.chess_board.board_array)
                    self.player1.update_q_value(state, action, 0, self.chess_board.board_array)
                    break

if __name__ == "__main__":
    game = Game()
    game.run(episodes=1000)
```

### Explanation

1. **ChessBoard Class**: Manages the game state and checks for a win or tie.
2. **Agent Class**: Implements the Q-learning algorithm.
   - `choose_action`: Selects an action based on the epsilon-greedy strategy.
   - `update_q_value`: Updates the Q-values based on the reward received.
3. **Game Class**: Manages the game loop and interactions between the agents.

### Next Steps

- **Train the Model**: Run the game for multiple episodes to allow the agents to learn.
- **Evaluate Performance**: After training, evaluate the performance of the agents by playing against them or comparing their win rates.
- **Refine the Model**: Experiment with different hyperparameters (epsilon, alpha, gamma) to improve learning.

This should give you a solid foundation for implementing RL in your Tic-Tac-Toe game.
