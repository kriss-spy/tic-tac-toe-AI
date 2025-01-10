import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm
import random
from collections import defaultdict


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)  # Flattened board for faster operations
        self.current_player = 1
        self.WINS = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.get_state()

    def get_state(self) -> tuple:
        return tuple(self.board)

    def make_move(self, position: int) -> bool:
        if self.board[position] == 0:
            self.board[position] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def get_valid_moves(self) -> list:
        return [i for i in range(9) if self.board[i] == 0]

    def is_winner(self, player: int) -> bool:
        return any(all(self.board[i] == player for i in win) for win in self.WINS)

    def is_draw(self) -> bool:
        return 0 not in self.board and not self.is_winner(1) and not self.is_winner(-1)

    def is_game_over(self) -> bool:
        return self.is_winner(1) or self.is_winner(-1) or self.is_draw()


class OptimizedQLearningAgent:
    def __init__(self, epsilon=0.3, alpha=0.2, gamma=0.95):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.999

    def get_action(self, state: tuple, valid_moves: list) -> int:
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        return max(valid_moves, key=lambda m: self.q_table[state][m])

    def learn(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_valid_moves: list,
    ):
        next_value = max(
            (self.q_table[next_state][m] for m in next_valid_moves), default=0
        )
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_value - current_q
        )

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class MinimaxExpert:
    def __init__(self):
        self.optimal_moves = {}
        self.WINS = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

    def build_cache(self):
        print("Precomputing optimal moves...")
        game = TicTacToe()
        self._minimax_cache(game, tuple(game.board))
        print("Done precomputing!")

    def _minimax_cache(self, game: TicTacToe, state: tuple) -> Tuple[int, int]:
        if game.is_winner(1):
            return 1, None
        if game.is_winner(-1):
            return -1, None
        if game.is_draw():
            return 0, None

        if state in self.optimal_moves:
            return self.optimal_moves[state]

        valid_moves = game.get_valid_moves()
        best_score = float("-inf") if game.current_player == 1 else float("inf")
        best_move = valid_moves[0]

        for move in valid_moves:
            game.make_move(move)
            score, _ = self._minimax_cache(game, tuple(game.board))
            game.board[move] = 0
            game.current_player = -game.current_player

            if game.current_player == 1:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        self.optimal_moves[state] = (best_score, best_move)
        return best_score, best_move

    def get_move(self, state: tuple) -> int:
        return self.optimal_moves[state][1]


def calculate_reward(game: TicTacToe, action: int, valid_moves: list) -> float:
    if game.is_winner(1):
        return 1.0
    if game.is_winner(-1):
        return -1.0
    if game.is_draw():
        return 0.5

    reward = 0.0
    # Check if we're blocking opponent's win
    temp_board = game.board.copy()
    for move in valid_moves:
        if move != action:
            temp_board[move] = -1
            if any(all(temp_board[i] == -1 for i in win) for win in game.WINS):
                reward += 0.2
            temp_board[move] = 0

    # Reward for controlling center
    if action == 4:
        reward += 0.1

    # Reward for creating winning opportunities
    temp_board = game.board.copy()
    winning_opportunities = 0
    for move in valid_moves:
        temp_board[move] = 1
        if any(all(temp_board[i] == 1 for i in win) for win in game.WINS):
            winning_opportunities += 1
        temp_board[move] = 0
    reward += 0.1 * winning_opportunities

    return reward


def evaluate_agent(agent, expert, n_games=100):
    optimal_moves = 0
    total_moves = 0
    results = {"wins": 0, "draws": 0}

    for _ in range(n_games):
        game = TicTacToe()
        while not game.is_game_over():
            state = game.get_state()
            valid_moves = game.get_valid_moves()

            if game.current_player == 1:
                agent_move = agent.get_action(state, valid_moves)
                expert_move = expert.get_move(state)
                optimal_moves += agent_move == expert_move
                total_moves += 1
                game.make_move(agent_move)
            else:
                expert_move = expert.get_move(state)
                game.make_move(expert_move)

        if game.is_winner(1):
            results["wins"] += 1
        elif game.is_draw():
            results["draws"] += 1

    return {
        "win_rate": results["wins"] / n_games,
        "draw_rate": results["draws"] / n_games,
        "optimal_rate": optimal_moves / total_moves if total_moves > 0 else 0,
    }


def train_and_evaluate(episodes=10000, eval_interval=500):
    expert = MinimaxExpert()
    expert.build_cache()

    agent = OptimizedQLearningAgent()
    metrics = {"win_rates": [], "draw_rates": [], "optimal_rates": [], "episodes": []}

    # Initial evaluation before training
    initial_metrics = evaluate_agent(agent, expert, n_games=100)
    print("\nInitial Performance (Before Training):")
    print(f"Win Rate: {initial_metrics['win_rate']:.1%}")
    print(f"Draw Rate: {initial_metrics['draw_rate']:.1%}")
    print(f"Optimal Move Rate: {initial_metrics['optimal_rate']:.1%}")

    # Training loop
    for episode in tqdm(range(episodes)):
        game = TicTacToe()
        while not game.is_game_over():
            state = game.get_state()
            valid_moves = game.get_valid_moves()

            if game.current_player == 1:
                action = agent.get_action(state, valid_moves)
                game.make_move(action)
                reward = calculate_reward(game, action, valid_moves)
                agent.learn(
                    state, action, reward, game.get_state(), game.get_valid_moves()
                )
                agent.decay_epsilon()
            else:
                expert_move = expert.get_move(state)
                game.make_move(expert_move)

        if (episode + 1) % eval_interval == 0:
            results = evaluate_agent(agent, expert)
            metrics["episodes"].append(episode + 1)
            metrics["win_rates"].append(results["win_rate"])
            metrics["draw_rates"].append(results["draw_rate"])
            metrics["optimal_rates"].append(results["optimal_rate"])

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["episodes"], metrics["win_rates"], "g-", label="Win Rate")
    plt.plot(metrics["episodes"], metrics["draw_rates"], "b-", label="Draw Rate")
    plt.plot(
        metrics["episodes"], metrics["optimal_rates"], "r-", label="Optimal Move Rate"
    )
    plt.title("Agent Performance vs Minimax Expert")
    plt.xlabel("Episodes")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final evaluation
    final_metrics = evaluate_agent(agent, expert, n_games=1000)
    print("\nFinal Performance (1000 games):")
    print(f"Win Rate: {final_metrics['win_rate']:.1%}")
    print(f"Draw Rate: {final_metrics['draw_rate']:.1%}")
    print(f"Optimal Move Rate: {final_metrics['optimal_rate']:.1%}")

    return agent


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    trained_agent = train_and_evaluate(episodes=10000, eval_interval=500)

# Precomputing optimal moves...
# Done precomputing!

# Initial Performance (Before Training):
# Win Rate: 0.0%
# Draw Rate: 12.0%
# Optimal Move Rate: 55.7%
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1703.94it/s]

# Final Performance (1000 games):
# Win Rate: 0.0%
# Draw Rate: 0.4%
# Optimal Move Rate: 0.8%
